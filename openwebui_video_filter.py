"""
title: Video Frames for Ollama Vision
version: 1.0.0
required_open_webui_version: 0.6.0
requirements: httpx
"""

from __future__ import annotations

import math
from pathlib import PurePath
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        frame_service_url: str = Field(
            default="http://host.docker.internal:8000",
            description="Base URL of the OCR backend's video frame service.",
        )
        context_size: int = Field(
            default=32768,
            ge=4096,
            description="Selected model context size in tokens.",
        )
        reserved_context_tokens: int = Field(
            default=8192,
            ge=0,
            description="Keep this much context for instructions, history, and the model reply.",
        )
        estimated_tokens_per_frame: int = Field(
            default=1536,
            ge=256,
            description="Conservative context estimate per resized video frame.",
        )
        max_frames: int = Field(default=24, ge=1, le=32)
        max_videos_per_message: int = Field(default=2, ge=1, le=8)
        sample_fps: float = Field(default=0.5, gt=0, le=2)
        max_frame_dimension: int = Field(default=768, ge=128, le=2048)
        max_video_mb: int = Field(default=200, ge=1, le=2048)
        request_timeout_seconds: float = Field(default=900, gt=0, le=3600)

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True

    @staticmethod
    def _is_video_file(item: dict[str, Any]) -> bool:
        metadata = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        content_type = str(
            item.get("content_type") or item.get("mime_type") or metadata.get("content_type") or ""
        ).lower()
        filename = str(item.get("name") or item.get("filename") or metadata.get("name") or "").lower()
        video_extensions = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv"}
        return content_type.startswith("video/") or PurePath(filename).suffix in video_extensions

    @staticmethod
    def _file_id(item: dict[str, Any]) -> str | None:
        value = item.get("id")
        if isinstance(value, str) and value and item.get("type", "file") in {"file", "video"}:
            return value
        return None

    @staticmethod
    def _text_from_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        return " ".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
        )

    def _estimate_existing_tokens(self, messages: list[dict[str, Any]]) -> int:
        text_characters = 0
        existing_images = 0
        for message in messages:
            content = message.get("content")
            text_characters += len(self._text_from_content(content))
            if isinstance(content, list):
                existing_images += sum(
                    1
                    for part in content
                    if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}
                )
        return math.ceil(text_characters / 2) + existing_images * self.valves.estimated_tokens_per_frame

    def _frame_budget(
        self,
        messages: list[dict[str, Any]],
        video_count: int,
        existing_image_count: int = 0,
    ) -> int:
        remaining = (
            self.valves.context_size
            - self.valves.reserved_context_tokens
            - self._estimate_existing_tokens(messages)
            - existing_image_count * self.valves.estimated_tokens_per_frame
        )
        context_limit = remaining // self.valves.estimated_tokens_per_frame
        budget = min(self.valves.max_frames, context_limit)
        if budget < video_count:
            raise ValueError(
                "Not enough estimated context remains for video frames. Start a new chat, "
                "shorten the conversation, or increase the filter's context budget."
            )
        return budget

    async def _read_openwebui_file(self, request: Any, file_id: str) -> bytes:
        if request is None:
            raise RuntimeError("Open WebUI request context is unavailable; cannot read the attached video.")

        file_url = request.app.url_path_for("get_file_content_by_id", id=file_id)
        headers = {}
        authorization = request.headers.get("authorization")
        if authorization:
            headers["authorization"] = authorization
        cookies = dict(request.cookies)
        transport = httpx.ASGITransport(app=request.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url=str(request.base_url),
            timeout=self.valves.request_timeout_seconds,
        ) as client:
            async with client.stream("GET", str(file_url), headers=headers, cookies=cookies) as response:
                response.raise_for_status()
                chunks = []
                total_bytes = 0
                async for chunk in response.aiter_bytes():
                    total_bytes += len(chunk)
                    if total_bytes > self.valves.max_video_mb * 1024 * 1024:
                        raise ValueError(f"Attached video exceeds {self.valves.max_video_mb} MB.")
                    chunks.append(chunk)
                return b"".join(chunks)

    async def _sample_frames(self, data: bytes, filename: str, frame_limit: int) -> list[dict[str, Any]]:
        url = f"{self.valves.frame_service_url.rstrip('/')}/v1/video/frames"
        async with httpx.AsyncClient(timeout=self.valves.request_timeout_seconds) as response:
            result = await response.post(
                url,
                params={
                    "max_frames": frame_limit,
                    "fps": self.valves.sample_fps,
                    "max_dimension": self.valves.max_frame_dimension,
                },
                content=data,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Filename": quote(filename),
                },
            )
            try:
                result.raise_for_status()
            except httpx.HTTPStatusError as exc:
                try:
                    detail = exc.response.json().get("detail")
                except (AttributeError, ValueError):
                    detail = exc.response.text
                if not isinstance(detail, str) or not detail:
                    detail = exc.response.reason_phrase
                raise RuntimeError(
                    f"Video frame service returned HTTP {exc.response.status_code}: {detail}"
                ) from exc
            payload = result.json()

        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            raise RuntimeError("The frame service returned no video frames.")
        return frames

    async def inlet(
        self,
        body: dict[str, Any],
        __request__=None,
        __user__: dict | None = None,
        __event_emitter__=None,
    ) -> dict[str, Any]:
        files = []
        existing_image_ids = set()
        seen_ids = set()
        metadata = body.get("metadata")
        metadata_files = metadata.get("files", []) if isinstance(metadata, dict) else []
        for collection in (metadata_files, body.get("files", [])):
            if not isinstance(collection, list):
                continue
            for item in collection:
                if not isinstance(item, dict) or not self._is_video_file(item):
                    if isinstance(item, dict):
                        metadata = item.get("meta") if isinstance(item.get("meta"), dict) else {}
                        content_type = str(
                            item.get("content_type") or item.get("mime_type") or metadata.get("content_type") or ""
                        ).lower()
                        image_id = item.get("id")
                        if content_type.startswith("image/") and isinstance(image_id, str):
                            existing_image_ids.add(image_id)
                    continue
                file_id = self._file_id(item)
                if file_id and file_id not in seen_ids:
                    seen_ids.add(file_id)
                    files.append((item, file_id))

        if not files:
            return body
        if len(files) > self.valves.max_videos_per_message:
            raise ValueError(f"Attach no more than {self.valves.max_videos_per_message} videos per message.")

        messages = body.get("messages") or []
        target_message = next((message for message in reversed(messages) if message.get("role") == "user"), None)
        if target_message is None:
            raise ValueError("Could not find the user message associated with the attached video.")

        total_frame_budget = self._frame_budget(messages, len(files), len(existing_image_ids))
        base_budget, remainder = divmod(total_frame_budget, len(files))
        image_parts = [
            {
                "type": "text",
                "text": "The following are timestamped frames sampled evenly across the attached video in chronological order.",
            }
        ]
        processed_ids = set()

        for video_index, (item, file_id) in enumerate(files):
            frame_limit = base_budget + (1 if video_index < remainder else 0)
            filename = str(item.get("name") or item.get("filename") or "video.mp4")
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": f"Sampling video frames: {filename}", "done": False}}
                )
            video_bytes = await self._read_openwebui_file(__request__, file_id)
            frames = await self._sample_frames(video_bytes, filename, frame_limit)
            for frame_index, frame in enumerate(frames, start=1):
                mime_type = frame.get("mime_type") or "image/jpeg"
                timestamp = float(frame.get("timestamp_seconds", 0.0))
                data_base64 = frame.get("data")
                if not isinstance(data_base64, str) or not data_base64:
                    raise RuntimeError("The frame service returned an invalid image frame.")
                image_url = f"data:{mime_type};base64,{data_base64}"
                image_parts.extend(
                    [
                        {
                            "type": "text",
                            "text": f"Video frame {frame_index}/{len(frames)} from {filename} at {timestamp:.2f}s:",
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]
                )
            processed_ids.add(file_id)

        content = target_message.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}] if content else []
        elif isinstance(content, list):
            content = list(content)
        elif content is None:
            content = []
        else:
            content = [{"type": "text", "text": str(content)}]
        content.extend(image_parts)
        target_message["content"] = content

        metadata = body.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("files"), list):
            metadata["files"] = [
                item for item in metadata["files"]
                if not (isinstance(item, dict) and item.get("id") in processed_ids)
            ]
        if isinstance(body.get("files"), list):
            body["files"] = [
                item for item in body["files"]
                if not (isinstance(item, dict) and item.get("id") in processed_ids)
            ]
        for message in messages:
            if isinstance(message.get("files"), list):
                message["files"] = [
                    item for item in message["files"]
                    if not (isinstance(item, dict) and item.get("id") in processed_ids)
                ]

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Added {len(image_parts) // 2} sampled frames to the model input",
                        "done": True,
                    },
                }
            )
        return body
