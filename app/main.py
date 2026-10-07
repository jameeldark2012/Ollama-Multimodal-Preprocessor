import asyncio
import base64
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import unquote

import httpx
from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.ocr import extract_attachment, image_from_data_url
from app.video import extract_video_frames

app = FastAPI(title="Ollama OCR", version="0.1.0")
logger = logging.getLogger("uvicorn.error")


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False


@app.get("/healthz")
async def health() -> dict[str, str]:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            if settings.backend == "llamacpp":
                response = await client.get(f"{settings.llamacpp_base_url}/health")
                response.raise_for_status()
                return {"status": "ok", "backend": "llamacpp", "model": settings.llamacpp_model}
            else:
                response = await client.get(f"{settings.ollama_base_url}/api/version")
                response.raise_for_status()
                return {"status": "ok", "backend": "ollama", "ollama_version": response.json().get("version", "unknown")}
    except httpx.HTTPError as exc:
        backend_label = "llama.cpp" if settings.backend == "llamacpp" else "Ollama"
        raise HTTPException(status_code=503, detail=f"{backend_label} is unavailable.") from exc


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": settings.public_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama-ocr",
            }
        ],
    }


@app.post("/v1/video/frames")
async def video_frames(
    request: Request,
    x_filename: str = Header(default="video.mp4"),
    max_frames: int = Query(default=settings.video_max_frames, ge=1, le=32),
    fps: float = Query(default=0.5, gt=0, le=2),
    max_dimension: int = Query(default=768, ge=128, le=2048),
) -> dict[str, Any]:
    filename = unquote(x_filename).replace("\\", "/").rsplit("/", 1)[-1] or "video.mp4"
    data = await request.body()
    if len(data) > settings.video_max_upload_bytes:
        raise HTTPException(status_code=413, detail="Video exceeds the configured upload limit.")

    try:
        duration, frames = await asyncio.to_thread(
            extract_video_frames,
            data,
            filename,
            fps=fps,
            min_frames=min(settings.video_min_frames, max_frames),
            max_frames=min(max_frames, settings.video_max_frames),
            max_dimension=min(max_dimension, settings.video_frame_max_dimension),
            jpeg_quality=settings.video_jpeg_quality,
            timeout_seconds=settings.video_ffmpeg_timeout_seconds,
            max_duration_seconds=settings.video_max_duration_seconds,
        )
    except ValueError as exc:
        logger.warning("Video rejected: file=%s reason=%s", filename, exc)
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if settings.ocr_debug:
        logger.info("Video sampled: file=%s duration=%.2fs frames=%d", filename, duration, len(frames))
    return {
        "filename": filename,
        "duration_seconds": duration,
        "frames": [
            {
                "timestamp_seconds": timestamp,
                "mime_type": "image/jpeg",
                "data": base64.b64encode(frame).decode("ascii"),
            }
            for timestamp, frame in frames
        ],
    }


@app.post("/v1/ocr")
async def ocr_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or "upload"
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the configured upload limit.")
    try:
        text, pages = await extract_attachment(filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except httpx.TimeoutException as exc:
        backend_label = "llama.cpp" if settings.backend == "llamacpp" else "Ollama"
        raise HTTPException(status_code=504, detail=f"{backend_label} timed out while processing the file.") from exc
    except httpx.HTTPError as exc:
        backend_label = "llama.cpp" if settings.backend == "llamacpp" else "Ollama"
        raise HTTPException(status_code=502, detail=f"{backend_label} failed to process the file.") from exc
    return {"filename": filename, "model": settings.active_model, "pages_processed": pages, "text": text}


@app.put("/process")
async def process_document(
    request: Request,
    x_filename: str = Header(default="upload"),
) -> dict[str, Any]:
    filename = unquote(x_filename).replace("\\", "/").rsplit("/", 1)[-1] or "upload"
    data = await request.body()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the configured upload limit.")
    try:
        text, pages = await extract_attachment(filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except httpx.TimeoutException as exc:
        backend_label = "llama.cpp" if settings.backend == "llamacpp" else "Ollama"
        raise HTTPException(status_code=504, detail=f"{backend_label} timed out while processing the file.") from exc
    except httpx.HTTPError as exc:
        backend_label = "llama.cpp" if settings.backend == "llamacpp" else "Ollama"
        raise HTTPException(status_code=502, detail=f"{backend_label} failed to process the file.") from exc

    return {
        "page_content": text,
        "metadata": {
            "source": filename,
            "file_name": filename,
            "file_content_type": request.headers.get("content-type", "application/octet-stream"),
            "pages_processed": pages,
        },
    }


def _extract_b64_from_data_url(url: str) -> tuple[str, str]:
    """Return (mime_type, base64_string) from a data URL without decoding the image bytes."""
    try:
        header, b64 = url.split(",", 1)
        if not header.startswith("data:") or ";base64" not in header:
            raise ValueError
        mime = header[len("data:"):header.index(";")]
        # Validate it is actually base64 without allocating the full decoded bytes
        import binascii
        binascii.a2b_base64(b64[:64] + "====")  # quick sanity check on first bytes only
        return mime, b64
    except (ValueError, Exception) as exc:
        raise ValueError("Images must be supplied as base64 data URLs.") from exc


async def _chat_completion(request: ChatRequest) -> tuple[str, str]:
    messages = []
    for message in request.messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        content = message.get("content", "")
        # image_entries stores (mime_type, base64_string) — never decoded to raw bytes
        image_entries: list[tuple[str, str]] = []
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url", "")
                    try:
                        mime, b64 = _extract_b64_from_data_url(image_url)
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=str(exc)) from exc
                    image_entries.append((mime, b64))
            content = "\n".join(text_parts)
        converted: dict[str, Any] = {"role": role, "content": str(content)}
        if image_entries:
            converted["image_entries"] = image_entries
        messages.append(converted)

    if not messages:
        raise HTTPException(status_code=400, detail="At least one supported chat message is required.")

    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
            if settings.backend == "llamacpp":
                # Build OpenAI-style content arrays, passing base64 strings directly
                openai_messages = []
                for msg in messages:
                    entries = msg.get("image_entries")
                    if entries:
                        parts: list[Any] = [{"type": "text", "text": msg["content"]}]
                        for mime, b64 in entries:
                            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
                        openai_messages.append({"role": msg["role"], "content": parts})
                    else:
                        openai_messages.append({"role": msg["role"], "content": msg["content"]})
                response = await client.post(
                    f"{settings.llamacpp_base_url}/chat/completions",
                    json={
                        "model": settings.llamacpp_model,
                        "messages": openai_messages,
                        "temperature": 0,
                        "max_tokens": settings.ollama_num_predict,
                    },
                )
                response.raise_for_status()
                reply = response.json()["choices"][0]["message"]["content"]
            else:
                # Convert image_entries tuples to plain base64 strings for Ollama's /api/chat format
                ollama_messages = []
                for msg in messages:
                    entries = msg.get("image_entries")
                    if entries:
                        ollama_msg: dict[str, Any] = {
                            "role": msg["role"],
                            "content": msg["content"],
                            "images": [b64 for _mime, b64 in entries],
                        }
                        ollama_messages.append(ollama_msg)
                    else:
                        ollama_messages.append({"role": msg["role"], "content": msg["content"]})
                response = await client.post(
                    f"{settings.ollama_base_url}/api/chat",
                    json={
                        "model": settings.ollama_model,
                        "messages": ollama_messages,
                        "stream": False,
                        "think": False,
                        "options": {"temperature": 0, "num_predict": settings.ollama_num_predict},
                    },
                )
                response.raise_for_status()
                reply = response.json()["message"]["content"]
    except httpx.TimeoutException as exc:
        backend_label = "llama.cpp" if settings.backend == "llamacpp" else "Ollama"
        raise HTTPException(status_code=504, detail=f"{backend_label} timed out while generating a response.") from exc
    except httpx.HTTPError as exc:
        backend_label = "llama.cpp" if settings.backend == "llamacpp" else "Ollama"
        raise HTTPException(status_code=502, detail=f"{backend_label} failed to generate a response.") from exc
    return reply, str(uuid.uuid4())


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest) -> Any:
    content, completion_id = await _chat_completion(request)
    result = {
        "id": f"chatcmpl-{completion_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": settings.public_model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }
    if not request.stream:
        return result

    async def stream() -> AsyncIterator[str]:
        for chunk in (
            {"id": result["id"], "object": "chat.completion.chunk", "created": result["created"], "model": result["model"], "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
            {"id": result["id"], "object": "chat.completion.chunk", "created": result["created"], "model": result["model"], "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]},
            {"id": result["id"], "object": "chat.completion.chunk", "created": result["created"], "model": result["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")