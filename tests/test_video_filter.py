from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
import pytest
from starlette.datastructures import Headers, URL

from openwebui_video_filter import Filter


class StubVideoFilter(Filter):
    async def _read_openwebui_file(self, request, file_id):
        assert file_id == "video-1"
        return b"video bytes"

    async def _sample_frames(self, data, filename, frame_limit):
        assert data == b"video bytes"
        assert filename == "clip.mp4"
        assert frame_limit == 2
        return [
            {"timestamp_seconds": 0.5, "mime_type": "image/jpeg", "data": "anBlZw=="},
            {"timestamp_seconds": 1.5, "mime_type": "image/jpeg", "data": "anBlZw=="},
        ]


@pytest.mark.asyncio
async def test_filter_injects_frames_and_preserves_document_references():
    video_ref = {
        "id": "video-1",
        "type": "file",
        "name": "clip.mp4",
        "content_type": "video/mp4",
    }
    pdf_ref = {
        "id": "pdf-1",
        "type": "file",
        "name": "notes.pdf",
        "content_type": "application/pdf",
    }
    body = {
        "model": "qwen-apex",
        "messages": [{"role": "user", "content": "Summarize this clip."}],
        "files": [video_ref, pdf_ref],
        "metadata": {"files": [video_ref, pdf_ref]},
    }
    filter_instance = StubVideoFilter()
    filter_instance.valves = Filter.Valves(
        context_size=32768,
        reserved_context_tokens=8192,
        estimated_tokens_per_frame=1536,
        max_frames=2,
    )

    result = await filter_instance.inlet(body, __request__=object())

    message = result["messages"][0]
    assert message["content"][0] == {"type": "text", "text": "Summarize this clip."}
    assert "chronological order" in message["content"][1]["text"]
    assert message["content"][2] == {
        "type": "text",
        "text": "Video frame 1/2 from clip.mp4 at 0.50s:",
    }
    assert message["content"][3] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,anBlZw=="},
    }
    assert result["files"] == [pdf_ref]
    assert result["metadata"]["files"] == [pdf_ref]
    assert result["model"] == "qwen-apex"

    original_content = list(message["content"])
    await filter_instance.inlet(body, __request__=object())
    assert message["content"] == original_content


@pytest.mark.asyncio
async def test_filter_ignores_non_video_files():
    body = {
        "messages": [{"role": "user", "content": "Read this."}],
        "metadata": {"files": [{"id": "pdf-1", "name": "notes.pdf", "content_type": "application/pdf"}]},
    }
    filter_instance = StubVideoFilter()

    result = await filter_instance.inlet(body, __request__=object())

    assert result is body
    assert result["messages"][0]["content"] == "Read this."


def test_frame_budget_shrinks_with_existing_conversation_context():
    filter_instance = Filter()
    filter_instance.valves = Filter.Valves(
        context_size=32768,
        reserved_context_tokens=8192,
        estimated_tokens_per_frame=1536,
        max_frames=8,
    )
    short_history = [{"role": "user", "content": "Look at this video."}]
    long_history = [{"role": "user", "content": "x" * 30000}]

    assert filter_instance._frame_budget(short_history, 1) == 8
    assert filter_instance._frame_budget(long_history, 1) < 8
    assert filter_instance._frame_budget(short_history, 1, existing_image_count=10) < 8


def test_frame_budget_rejects_context_with_no_video_room():
    filter_instance = Filter()
    filter_instance.valves = Filter.Valves(
        context_size=4096,
        reserved_context_tokens=4096,
        estimated_tokens_per_frame=1536,
        max_frames=8,
    )

    with pytest.raises(ValueError, match="Not enough estimated context"):
        filter_instance._frame_budget([], 1)


@pytest.mark.asyncio
async def test_filter_reads_video_via_openwebui_file_route_with_user_auth():
    webui = FastAPI()

    @webui.get("/api/v1/files/{id}/content", name="get_file_content_by_id")
    async def get_file_content(id: str, request: Request):
        if request.headers.get("authorization") != "Bearer test-user-token":
            raise HTTPException(status_code=401)
        return Response(content=f"video:{id}".encode())

    request = SimpleNamespace(
        app=webui,
        base_url=URL("http://openwebui.local/"),
        headers=Headers({"authorization": "Bearer test-user-token"}),
        cookies={},
    )

    result = await Filter()._read_openwebui_file(request, "video-1")

    assert result == b"video:video-1"


@pytest.mark.asyncio
async def test_frame_service_error_includes_backend_detail(monkeypatch):
    import httpx

    class ErrorResponse:
        status_code = 415
        reason_phrase = "Unsupported Media Type"

        def json(self):
            return {"detail": "The uploaded video could not be opened."}

        @property
        def text(self):
            return ""

        def raise_for_status(self):
            request = httpx.Request("POST", "http://frame-service/v1/video/frames")
            response = httpx.Response(self.status_code, request=request, json=self.json())
            raise httpx.HTTPStatusError("status error", request=request, response=response)

    class ErrorClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            return ErrorResponse()

    monkeypatch.setattr("openwebui_video_filter.httpx.AsyncClient", ErrorClient)

    with pytest.raises(RuntimeError, match="HTTP 415: The uploaded video could not be opened"):
        await Filter()._sample_frames(b"invalid video", "clip.mkv", 2)
