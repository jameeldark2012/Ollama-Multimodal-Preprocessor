"""
title: Video Frames for Ollama Vision
version: 1.1.0
required_open_webui_version: 0.6.0
requirements: httpx
"""

from __future__ import annotations

import base64
import importlib
import math
from collections import Counter
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

    @staticmethod
    def _remove_processed_files(body: dict[str, Any], processed_ids: set[str]) -> None:
        if not processed_ids:
            return

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
        messages = body.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict) or not isinstance(message.get("files"), list):
                    continue
                message["files"] = [
                    item for item in message["files"]
                    if not (isinstance(item, dict) and item.get("id") in processed_ids)
                ]

    async def _find_persisted_video_frames(
        self,
        chat_id: str,
        video_file_id: str,
        user_id: str,
        current_message_id: str,
    ) -> tuple[str, list[dict[str, Any]]] | None:
        chats = importlib.import_module("open_webui.models.chats").Chats
        if not await chats.get_chat_by_id_and_user_id(chat_id, user_id):
            return None

        messages_map = await chats.get_messages_map_by_chat_id(chat_id) or {}
        get_message_list = importlib.import_module("open_webui.utils.misc").get_message_list
        active_messages = get_message_list(messages_map, current_message_id)
        for message in active_messages:
            message_id = message.get("id")
            if message.get("role") != "user":
                continue
            frame_files = [
                file
                for file in message.get("files", [])
                if isinstance(file, dict)
                and file.get("video_frame_source_id") == video_file_id
            ]
            if frame_files:
                return message_id, frame_files
        return None

    async def _request_user_message_index(
        self,
        chat_id: str,
        user_id: str,
        current_message_id: str,
    ) -> int | None:
        chats = importlib.import_module("open_webui.models.chats").Chats
        if not await chats.get_chat_by_id_and_user_id(chat_id, user_id):
            return None

        messages_map = await chats.get_messages_map_by_chat_id(chat_id) or {}
        get_message_list = importlib.import_module("open_webui.utils.misc").get_message_list
        user_messages = [
            message
            for message in get_message_list(messages_map, current_message_id)
            if message.get("role") == "user"
        ]
        return next(
            (index for index, message in enumerate(user_messages) if message.get("id") == current_message_id),
            None,
        )

    async def _restore_persisted_video_frames(
        self,
        request: Any,
        chat_id: str,
        user_id: str,
        current_message_id: str,
        messages: list[dict[str, Any]],
        source_message_id: str,
        filename: str,
        frame_files: list[dict[str, Any]],
    ) -> int:
        user_message_index = await self._request_user_message_index(
            chat_id,
            user_id,
            source_message_id,
        )
        user_messages = [message for message in messages if message.get("role") == "user"]
        if user_message_index is None or user_message_index >= len(user_messages):
            raise RuntimeError("Could not restore saved video frames to their original user message.")
        target_message = user_messages[user_message_index]

        content = target_message.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}] if content else []
        elif isinstance(content, list):
            content = list(content)
        elif content is None:
            content = []
        else:
            content = [{"type": "text", "text": str(content)}]

        restored_count = 0
        existing_image_urls = Counter()
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {"image_url", "input_image"}:
                continue
            image_payload = part.get("image_url")
            existing_url = image_payload.get("url") if isinstance(image_payload, dict) else image_payload
            if isinstance(existing_url, str) and existing_url.startswith("data:image/"):
                existing_image_urls[existing_url] += 1

        for frame_file in frame_files:
            frame_file_id = frame_file.get("id")
            if not isinstance(frame_file_id, str) or not frame_file_id:
                raise RuntimeError("A saved video frame is missing its Open WebUI file ID.")
            image_bytes = await self._read_openwebui_file(request, frame_file_id)
            mime_type = str(frame_file.get("content_type") or "image/jpeg")
            image_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
            stored_url = frame_file.get("url")

            if existing_image_urls[image_url]:
                existing_image_urls[image_url] -= 1
                continue

            stored_part = next(
                (
                    part
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") in {"image_url", "input_image"}
                    and (
                        (part.get("image_url", {}).get("url") if isinstance(part.get("image_url"), dict)
                         else part.get("image_url"))
                        == stored_url
                    )
                ),
                None,
            ) if stored_url else None
            if stored_part is not None:
                image_payload = stored_part.get("image_url")
                if isinstance(image_payload, dict):
                    stored_part["image_url"] = {**image_payload, "url": image_url}
                else:
                    stored_part["image_url"] = image_url
                continue

            frame_index = frame_file.get("video_frame_index", restored_count + 1)
            timestamp = float(frame_file.get("video_frame_timestamp", 0.0))
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Video frame {frame_index} from {filename} at {timestamp:.2f}s:",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            )
            restored_count += 1

        target_message["content"] = content
        return restored_count

    async def _persist_video_frames(
        self,
        request: Any,
        user_data: dict[str, Any],
        chat_id: str,
        message_id: str,
        video_file_id: str,
        filename: str,
        frames: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        chats = importlib.import_module("open_webui.models.chats").Chats
        users = importlib.import_module("open_webui.models.users").Users
        upload_image = importlib.import_module("open_webui.routers.images").upload_image

        user_id = user_data.get("id")
        user = await users.get_user_by_id(user_id) if user_id else None
        if user is None or not await chats.get_chat_by_id_and_user_id(chat_id, user_id):
            raise RuntimeError("Could not verify chat ownership to save sampled video frames.")

        message = await chats.get_message_by_id_and_message_id(chat_id, message_id)
        if not message or message.get("role") != "user":
            raise RuntimeError("Could not find the user message that owns the attached video.")

        existing_files = message.get("files") or []
        if any(
            isinstance(file, dict) and file.get("video_frame_source_id") == video_file_id
            for file in existing_files
        ):
            return [
                file
                for file in existing_files
                if isinstance(file, dict) and file.get("video_frame_source_id") == video_file_id
            ]

        persisted_files = []
        for frame_index, frame in enumerate(frames, start=1):
            mime_type = str(frame.get("mime_type") or "image/jpeg")
            encoded_data = frame.get("data")
            if not isinstance(encoded_data, str) or not encoded_data:
                raise RuntimeError("The frame service returned an invalid image frame.")
            try:
                image_bytes = base64.b64decode(encoded_data, validate=True)
            except ValueError as exc:
                raise RuntimeError("The frame service returned invalid base64 image data.") from exc

            timestamp = float(frame.get("timestamp_seconds", 0.0))
            _, image_file = await upload_image(
                request,
                image_bytes,
                mime_type,
                {
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "video_frame_source_id": video_file_id,
                    "video_frame_index": frame_index,
                    "video_frame_timestamp": timestamp,
                },
                user,
            )
            persisted_files.append(
                {
                    **image_file,
                    "type": "image",
                    "name": f"{filename} frame {frame_index:02d} at {timestamp:.2f}s",
                    "video_frame_source_id": video_file_id,
                    "video_frame_index": frame_index,
                    "video_frame_timestamp": timestamp,
                }
            )

        await chats.upsert_message_to_chat_by_id_and_message_id(
            chat_id,
            message_id,
            {"files": [*existing_files, *persisted_files]},
            touch=False,
        )
        return persisted_files

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
        __metadata__: dict | None = None,
    ) -> dict[str, Any]:
        files = []
        existing_image_ids = set()
        processed_ids = set()
        persisted_frames_to_restore = []
        seen_ids = set()
        metadata = body.get("metadata")
        request_metadata = __metadata__ if isinstance(__metadata__, dict) else {}
        chat_id_value = request_metadata.get("chat_id")
        chat_id_value = chat_id_value or (metadata.get("chat_id") if isinstance(metadata, dict) else None)
        chat_id_value = chat_id_value or body.get("chat_id")
        chat_id = str(chat_id_value) if chat_id_value is not None else None
        user_message_id = request_metadata.get("user_message_id")
        user_id = __user__.get("id") if isinstance(__user__, dict) else None
        metadata_files = metadata.get("files", []) if isinstance(metadata, dict) else []
        for collection in (metadata_files, body.get("files", [])):
            if not isinstance(collection, list):
                continue
            for item in collection:
                if not isinstance(item, dict) or not self._is_video_file(item):
                    if isinstance(item, dict):
                        item_metadata = item.get("meta") if isinstance(item.get("meta"), dict) else {}
                        content_type = str(
                            item.get("content_type")
                            or item.get("mime_type")
                            or item_metadata.get("content_type")
                            or ""
                        ).lower()
                        image_id = item.get("id")
                        if content_type.startswith("image/") and isinstance(image_id, str):
                            existing_image_ids.add(image_id)
                    continue
                file_id = self._file_id(item)
                if not file_id or file_id in seen_ids:
                    continue
                seen_ids.add(file_id)
                if chat_id and user_id:
                    current_message_id = str(user_message_id) if user_message_id else ""
                    persisted = (
                        await self._find_persisted_video_frames(
                            chat_id,
                            file_id,
                            user_id,
                            current_message_id,
                        )
                        if current_message_id
                        else None
                    )
                else:
                    persisted = None
                if persisted:
                    processed_ids.add(file_id)
                    persisted_frames_to_restore.append(
                        (item, persisted[0], persisted[1])
                    )
                    continue
                files.append((item, file_id))

        messages = body.get("messages") or []
        restored_frame_count = 0
        for item, source_message_id, frame_files in persisted_frames_to_restore:
            if not chat_id or not user_id or not user_message_id:
                raise RuntimeError("Chat identity is required to restore persisted video frames.")
            restored_frame_count += await self._restore_persisted_video_frames(
                __request__,
                chat_id,
                user_id,
                str(user_message_id),
                messages,
                source_message_id,
                str(item.get("name") or item.get("filename") or "video.mp4"),
                frame_files,
            )

        if not files:
            self._remove_processed_files(body, processed_ids)
            if restored_frame_count and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Restored {restored_frame_count} saved video frames to their original message",
                            "done": True,
                        },
                    }
                )
            return body
        if len(files) > self.valves.max_videos_per_message:
            raise ValueError(f"Attach no more than {self.valves.max_videos_per_message} videos per message.")

        target_message = next(
            (
                message
                for message in messages
                if message.get("role") == "user"
                and user_message_id
                and message.get("id") == user_message_id
            ),
            None,
        )
        if target_message is None and chat_id and user_id and user_message_id:
            user_messages = [message for message in messages if message.get("role") == "user"]
            user_message_index = await self._request_user_message_index(
                chat_id,
                user_id,
                str(user_message_id),
            )
            if user_message_index is not None and user_message_index < len(user_messages):
                target_message = user_messages[user_message_index]
            else:
                raise RuntimeError("Could not map the saved user message to the model request history.")
        if target_message is None:
            target_message = next(
                (message for message in reversed(messages) if message.get("role") == "user"),
                None,
            )
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
        for video_index, (item, file_id) in enumerate(files):
            frame_limit = base_budget + (1 if video_index < remainder else 0)
            filename = str(item.get("name") or item.get("filename") or "video.mp4")
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": f"Sampling video frames: {filename}", "done": False}}
                )
            video_bytes = await self._read_openwebui_file(__request__, file_id)
            frames = await self._sample_frames(video_bytes, filename, frame_limit)
            if chat_id and user_message_id:
                if not isinstance(__user__, dict):
                    raise RuntimeError("Open WebUI user context is required to save sampled video frames.")
                await self._persist_video_frames(
                    __request__,
                    __user__,
                    chat_id,
                    user_message_id,
                    file_id,
                    filename,
                    frames,
                )
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

        self._remove_processed_files(body, processed_ids)

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
