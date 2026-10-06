from copy import deepcopy
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
import pytest
from starlette.datastructures import Headers, URL

from openwebui_video_filter import Filter


class StubVideoFilter(Filter):
    def __init__(self, persisted_frames=None):
        super().__init__()
        self.persisted_frames = persisted_frames if persisted_frames is not None else {}
        self.sample_calls = 0

    async def _read_openwebui_file(self, request, file_id):
        if file_id == "video-1":
            return b"video bytes"
        if file_id.startswith("frame-"):
            return b"jpeg"
        raise AssertionError(f"Unexpected Open WebUI file ID: {file_id}")

    async def _sample_frames(self, data, filename, frame_limit):
        self.sample_calls += 1
        assert data == b"video bytes"
        assert filename == "clip.mp4"
        assert frame_limit == 2
        return [
            {"timestamp_seconds": 0.5, "mime_type": "image/jpeg", "data": "anBlZw=="},
            {"timestamp_seconds": 1.5, "mime_type": "image/jpeg", "data": "anBlZw=="},
        ]

    async def _find_persisted_video_frames(self, chat_id, video_file_id, user_id, current_message_id):
        return self.persisted_frames.get((chat_id, video_file_id))

    async def _request_user_message_index(self, chat_id, user_id, current_message_id):
        return {"user-1": 0, "user-2": 1, "user-3": 0}.get(current_message_id)

    async def _persist_video_frames(
        self, request, user_data, chat_id, message_id, video_file_id, filename, frames
    ):
        attachments = [
            {
                "id": f"frame-{frame_index}",
                "url": f"/api/v1/files/frame-{frame_index}/content",
                "name": f"{filename} frame {frame_index:02d}",
                "type": "image",
                "content_type": "image/jpeg",
                "video_frame_source_id": video_file_id,
                "video_frame_index": frame_index,
            }
            for frame_index, _ in enumerate(frames, start=1)
        ]
        self.persisted_frames[(chat_id, video_file_id)] = (message_id, attachments)
        return attachments


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


@pytest.mark.asyncio
async def test_filter_persists_frames_on_original_turn_and_reuses_them_after_restart():
    persisted_frames = {}
    filter_instance = StubVideoFilter(persisted_frames)
    filter_instance.valves = Filter.Valves(
        context_size=32768,
        reserved_context_tokens=8192,
        estimated_tokens_per_frame=1536,
        max_frames=2,
    )
    video_ref = {
        "id": "video-1",
        "type": "file",
        "name": "clip.mp4",
        "content_type": "video/mp4",
    }

    def make_body(chat_id, message_id, text, messages=None):
        return {
            "messages": messages or [{"role": "user", "content": text}],
            "files": [video_ref.copy()],
            "metadata": {"chat_id": chat_id, "files": [video_ref.copy()]},
        }

    first_chat_body = make_body("chat-1", "user-1", "Summarize this clip.")
    await filter_instance.inlet(
        first_chat_body,
        __request__=object(),
        __user__={"id": "user-owner"},
        __metadata__={"chat_id": "chat-1", "user_message_id": "user-1"},
    )
    original_turn = first_chat_body["messages"][0]
    assert any(
        part.get("type") == "image_url"
        for part in original_turn["content"]
        if isinstance(part, dict)
    )
    assert len(persisted_frames[("chat-1", "video-1")][1]) == 2
    assert filter_instance.sample_calls == 1

    followup_messages = [
        {"role": "user", "content": "Summarize this clip."},
        {"role": "assistant", "content": "Summary."},
        {"role": "user", "content": "Thanks."},
    ]
    followup_body = make_body("chat-1", "user-2", "Thanks.", followup_messages)
    restarted_filter = StubVideoFilter(persisted_frames)
    restarted_filter.valves = filter_instance.valves
    await restarted_filter.inlet(
        followup_body,
        __request__=object(),
        __user__={"id": "user-owner"},
        __metadata__={"chat_id": "chat-1", "user_message_id": "user-2"},
    )

    assert restarted_filter.sample_calls == 0
    assert followup_body["files"] == []
    assert followup_body["metadata"]["files"] == []
    assert any(
        part.get("type") == "image_url"
        for part in followup_body["messages"][0]["content"]
        if isinstance(part, dict)
    )
    assert sum(
        part.get("type") == "image_url"
        for part in followup_body["messages"][0]["content"]
        if isinstance(part, dict)
    ) == 2
    assert followup_body["messages"][2]["content"] == "Thanks."

    already_restored_messages = [
        {"role": "user", "content": deepcopy(original_turn["content"])},
        {"role": "assistant", "content": "Summary."},
        {"role": "user", "content": "One more question."},
    ]
    already_restored_body = make_body(
        "chat-1",
        "user-4",
        "One more question.",
        already_restored_messages,
    )
    await restarted_filter.inlet(
        already_restored_body,
        __request__=object(),
        __user__={"id": "user-owner"},
        __metadata__={"chat_id": "chat-1", "user_message_id": "user-4"},
    )

    assert sum(
        part.get("type") == "image_url"
        for part in already_restored_body["messages"][0]["content"]
        if isinstance(part, dict)
    ) == 2
    assert restarted_filter.sample_calls == 0

    another_chat_body = make_body("chat-2", "user-3", "Summarize this clip.")
    await restarted_filter.inlet(
        another_chat_body,
        __request__=object(),
        __user__={"id": "user-owner"},
        __metadata__={"chat_id": "chat-2", "user_message_id": "user-3"},
    )

    assert restarted_filter.sample_calls == 1


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
async def test_persist_video_frames_uploads_and_attaches_images_to_source_message(monkeypatch):
    video_ref = {"id": "video-1", "type": "file", "name": "clip.mp4"}
    source_message = {"id": "user-1", "role": "user", "files": [video_ref]}
    uploaded = []

    class FakeChats:
        async def get_chat_by_id_and_user_id(self, chat_id, user_id):
            assert (chat_id, user_id) == ("chat-1", "owner-1")
            return object()

        async def get_message_by_id_and_message_id(self, chat_id, message_id):
            assert (chat_id, message_id) == ("chat-1", "user-1")
            return source_message

        async def upsert_message_to_chat_by_id_and_message_id(
            self, chat_id, message_id, changes, *, touch
        ):
            assert (chat_id, message_id, touch) == ("chat-1", "user-1", False)
            source_message.update(changes)

    class FakeUsers:
        async def get_user_by_id(self, user_id):
            assert user_id == "owner-1"
            return SimpleNamespace(id=user_id)

    async def fake_upload_image(request, image_bytes, mime_type, metadata, user):
        assert image_bytes == b"jpeg"
        assert mime_type == "image/jpeg"
        assert metadata["message_id"] == "user-1"
        assert metadata["video_frame_source_id"] == "video-1"
        assert user.id == "owner-1"
        file_index = len(uploaded) + 1
        uploaded.append(metadata)
        return None, {
            "id": f"frame-{file_index}",
            "url": f"/api/v1/files/frame-{file_index}/content",
            "name": "generated-image.jpg",
            "content_type": mime_type,
        }

    modules = {
        "open_webui.models.chats": SimpleNamespace(Chats=FakeChats()),
        "open_webui.models.users": SimpleNamespace(Users=FakeUsers()),
        "open_webui.routers.images": SimpleNamespace(upload_image=fake_upload_image),
    }
    monkeypatch.setattr("openwebui_video_filter.importlib.import_module", modules.__getitem__)

    attachments = await Filter()._persist_video_frames(
        request=object(),
        user_data={"id": "owner-1"},
        chat_id="chat-1",
        message_id="user-1",
        video_file_id="video-1",
        filename="clip.mp4",
        frames=[
            {"timestamp_seconds": 0.5, "mime_type": "image/jpeg", "data": "anBlZw=="},
            {"timestamp_seconds": 1.5, "mime_type": "image/jpeg", "data": "anBlZw=="},
        ],
    )

    assert len(uploaded) == 2
    assert source_message["files"][0] == video_ref
    assert [file["video_frame_index"] for file in source_message["files"][1:]] == [1, 2]
    assert all(file["type"] == "image" for file in attachments)


@pytest.mark.asyncio
async def test_find_persisted_frames_only_matches_active_chat_branch(monkeypatch):
    messages_map = {
        "user-1": {
            "id": "user-1",
            "role": "user",
            "parentId": None,
            "files": [{"id": "frame-active", "video_frame_source_id": "video-1"}],
        },
        "assistant-1": {"id": "assistant-1", "role": "assistant", "parentId": "user-1"},
        "user-2": {"id": "user-2", "role": "user", "parentId": "assistant-1"},
        "user-branch": {
            "id": "user-branch",
            "role": "user",
            "parentId": "user-1",
            "files": [{"id": "frame-branch", "video_frame_source_id": "video-1"}],
        },
    }

    class FakeChats:
        async def get_chat_by_id_and_user_id(self, chat_id, user_id):
            return object() if (chat_id, user_id) == ("chat-1", "owner-1") else None

        async def get_messages_map_by_chat_id(self, chat_id):
            return messages_map

    def get_message_list(message_map, current_message_id):
        path = []
        while current_message_id:
            message = message_map[current_message_id]
            path.append(message)
            current_message_id = message.get("parentId")
        return list(reversed(path))

    modules = {
        "open_webui.models.chats": SimpleNamespace(Chats=FakeChats()),
        "open_webui.utils.misc": SimpleNamespace(get_message_list=get_message_list),
    }
    monkeypatch.setattr("openwebui_video_filter.importlib.import_module", modules.__getitem__)

    result = await Filter()._find_persisted_video_frames(
        "chat-1", "video-1", "owner-1", "user-2"
    )

    assert result == ("user-1", messages_map["user-1"]["files"])


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
