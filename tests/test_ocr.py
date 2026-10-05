from dataclasses import replace
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from app import ocr
from app.config import settings
from app.main import app
from app import video


@pytest.mark.asyncio
async def test_text_file_is_returned_without_model_call():
    text = "بسم الله الرحمن الرحيم"

    result, pages = await ocr.extract_attachment("sample.txt", text.encode("utf-8"))

    assert result == text
    assert pages is None


@pytest.mark.asyncio
async def test_json_is_returned_unchanged():
    source = b'{"name":"Noor","value":3}'

    result, pages = await ocr.extract_attachment("sample.json", source)

    assert result == source.decode()
    assert pages is None


@pytest.mark.asyncio
async def test_pdf_renders_each_page_for_vision_model(monkeypatch, caplog):
    document = fitz.open()
    document.new_page()
    document.new_page()
    pdf_bytes = document.tobytes()
    document.close()
    calls = []

    async def fake_ask(client, image_bytes):
        calls.append(image_bytes)
        return f"page {len(calls)}"

    monkeypatch.setattr(ocr, "_ask_ollama", fake_ask)
    monkeypatch.setattr(ocr, "settings", replace(ocr.settings, ocr_debug=True))

    with caplog.at_level("INFO", logger="uvicorn.error"):
        text, pages = await ocr.extract_attachment("sample.pdf", pdf_bytes)

    assert text == "page 1\n\npage 2"
    assert pages == 2
    assert len(calls) == 2
    assert "PDF page completed: file=sample.pdf page=1/2" in caplog.text
    assert "PDF page completed: file=sample.pdf page=2/2" in caplog.text
    assert "OCR finished: file=sample.pdf pages=2" in caplog.text


@pytest.mark.asyncio
async def test_rejects_unknown_file_type():
    with pytest.raises(ValueError, match="Unsupported file type"):
        await ocr.extract_attachment("sample.exe", b"not an OCR document")


def test_decodes_inline_image_data_url():
    assert ocr.image_from_data_url("data:image/png;base64,aGVsbG8=") == b"hello"


def test_rejects_remote_image_url():
    with pytest.raises(ValueError, match="base64 data URLs"):
        ocr.image_from_data_url("https://example.com/image.png")


def test_external_document_loader_process_endpoint(monkeypatch):
    async def fake_extract_attachment(filename, data):
        assert filename == "sample scan.pdf"
        assert data == b"pdf bytes"
        return "extracted text", 1

    monkeypatch.setattr("app.main.extract_attachment", fake_extract_attachment)

    with TestClient(app) as client:
        response = client.put(
            "/process",
            content=b"pdf bytes",
            headers={"Content-Type": "application/pdf", "X-Filename": "sample%20scan.pdf"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "page_content": "extracted text",
        "metadata": {
            "source": "sample scan.pdf",
            "file_name": "sample scan.pdf",
            "file_content_type": "application/pdf",
            "pages_processed": 1,
        },
    }


def test_video_sampler_spreads_frames_across_duration_and_obeys_cap(monkeypatch, tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    monkeypatch.setattr(video.shutil, "which", lambda command: command)
    monkeypatch.setattr(video, "_probe_duration", lambda path: 20.0)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        output_path = Path(command[-1])
        output_path.write_bytes(b"jpeg frame")
        return video.subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(video.subprocess, "run", fake_run)

    duration, frames = video.sample_video_frames(
        video_path,
        fps=0.5,
        min_frames=4,
        max_frames=4,
        max_dimension=768,
        jpeg_quality=5,
        timeout_seconds=10,
    )

    assert duration == 20.0
    assert [timestamp for timestamp, _ in frames] == [2.5, 7.5, 12.5, 17.5]
    assert all(frame_bytes == b"jpeg frame" for _, frame_bytes in frames)
    assert all("scale=768:768:force_original_aspect_ratio=decrease" in command for command in commands)


def test_video_sampler_keeps_short_clip_representative(monkeypatch, tmp_path):
    video_path = tmp_path / "short.mp4"
    video_path.write_bytes(b"video")
    monkeypatch.setattr(video.shutil, "which", lambda command: command)
    monkeypatch.setattr(video, "_probe_duration", lambda path: 2.0)

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"jpeg frame")
        return video.subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(video.subprocess, "run", fake_run)

    duration, frames = video.sample_video_frames(
        video_path,
        fps=0.5,
        min_frames=4,
        max_frames=8,
        max_dimension=768,
        jpeg_quality=5,
        timeout_seconds=10,
    )

    assert duration == 2.0
    assert [timestamp for timestamp, _ in frames] == [0.25, 0.75, 1.25, 1.75]


def test_video_sampler_requires_ffmpeg(monkeypatch, tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    monkeypatch.setattr(video.shutil, "which", lambda command: None)

    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        video.sample_video_frames(
            video_path,
            fps=0.5,
            min_frames=4,
            max_frames=4,
            max_dimension=768,
            jpeg_quality=5,
            timeout_seconds=10,
        )


def test_rejects_unsupported_video_extension():
    with pytest.raises(ValueError, match="Unsupported video type"):
        video.extract_video_frames(
            b"video",
            "sample.pdf",
            fps=0.5,
            min_frames=4,
            max_frames=4,
            max_dimension=768,
            jpeg_quality=5,
            timeout_seconds=10,
            max_duration_seconds=300,
        )


def test_video_frames_endpoint_returns_images_not_document_text(monkeypatch):
    monkeypatch.setattr(
        "app.main.extract_video_frames",
        lambda data, filename, **kwargs: (2.0, [(0.5, b"jpeg-one"), (1.5, b"jpeg-two")]),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/video/frames?max_frames=2&fps=1&max_dimension=512",
            content=b"video bytes",
            headers={"Content-Type": "application/octet-stream", "X-Filename": "sample.mp4"},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["duration_seconds"] == 2.0
    assert result["frames"] == [
        {"timestamp_seconds": 0.5, "mime_type": "image/jpeg", "data": "anBlZy1vbmU="},
        {"timestamp_seconds": 1.5, "mime_type": "image/jpeg", "data": "anBlZy10d28="},
    ]
    assert "page_content" not in result


def test_video_frames_endpoint_defaults_to_configured_frame_cap(monkeypatch):
    extracted = {}

    def fake_extract(data, filename, **kwargs):
        extracted.update(kwargs)
        return 2.0, [(0.5, b"jpeg-one")]

    monkeypatch.setattr("app.main.settings", replace(settings, video_max_frames=24))
    monkeypatch.setattr("app.main.extract_video_frames", fake_extract)

    with TestClient(app) as client:
        response = client.post(
            "/v1/video/frames",
            content=b"video bytes",
            headers={"Content-Type": "application/octet-stream", "X-Filename": "sample.mkv"},
        )

    assert response.status_code == 200
    assert extracted["max_frames"] == 24


def test_video_frames_endpoint_logs_rejection_reason(monkeypatch, caplog):
    def reject_video(data, filename, **kwargs):
        raise ValueError("The uploaded video could not be opened.")

    monkeypatch.setattr("app.main.extract_video_frames", reject_video)

    with TestClient(app) as client:
        with caplog.at_level("WARNING", logger="uvicorn.error"):
            response = client.post(
                "/v1/video/frames",
                content=b"video bytes",
                headers={"Content-Type": "application/octet-stream", "X-Filename": "sample.mkv"},
            )

    assert response.status_code == 415
    assert "Video rejected: file=sample.mkv reason=The uploaded video could not be opened." in caplog.text