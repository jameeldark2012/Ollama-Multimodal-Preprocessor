"""Pure in-memory simulations for the async OCR and Gemini quota paths."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import fitz
import pytest
from PIL import Image

from app import ocr
from app.gemini_backend import GeminiOCRBackend, GeminiRequestTimeout
from app.rate_limiter import RateLimiter


class MemoryCheckpointManager:
    """Checkpoint double that proves all persistence calls are awaited."""

    def __init__(self) -> None:
        self.saved_pages: dict[int, str] = {}
        self.progress: list[tuple[int, int, str]] = []

    async def calculate_file_hash(self, data: bytes) -> str:
        return "simulation"

    async def find_checkpoint(self, file_hash: str):
        return None

    async def create_checkpoint(self, file_hash: str, filename: str, total_pages: int) -> Path:
        return Path("memory-checkpoint")

    async def load_failed_pages(self, checkpoint_dir: Path) -> dict[int, dict]:
        return {}

    async def save_page(self, checkpoint_dir: Path, page_number: int, text: str) -> None:
        self.saved_pages[page_number] = text

    async def save_failed_pages(self, checkpoint_dir: Path, failed_pages: dict[int, dict]) -> None:
        return None

    async def update_progress(self, checkpoint_dir: Path, completed: int, total: int, status: str) -> None:
        self.progress.append((completed, total, status))


@pytest.mark.asyncio
async def test_pdf_gemini_flow_is_async_and_uses_only_memory_doubles(monkeypatch):
    document = fitz.open()
    document.new_page()
    document.new_page()
    pdf_bytes = document.tobytes()
    document.close()

    checkpoint = MemoryCheckpointManager()
    requests: list[int] = []

    class SimulatedGemini:
        batch_size = 2

        async def transcribe_batch(self, images: list[bytes]):
            requests.append(len(images))
            return [(f"simulated page {index + 1}", None) for index in range(len(images))]

    monkeypatch.setattr(ocr, "checkpoint_manager", checkpoint)
    monkeypatch.setattr(ocr, "gemini_backend", SimulatedGemini())
    monkeypatch.setattr(ocr, "settings", replace(ocr.settings, backend="gemini", ocr_debug=False))

    text, page_count = await ocr.extract_attachment("simulation.pdf", pdf_bytes)

    assert text == "simulated page 1\n\nsimulated page 2"
    assert page_count == 2
    assert requests == [2]
    assert checkpoint.saved_pages == {1: "simulated page 1", 2: "simulated page 2"}
    assert checkpoint.progress[-1] == (2, 2, "completed")


@pytest.mark.asyncio
async def test_gemini_batch_is_split_before_effective_tpm_is_exceeded(monkeypatch):
    backend = object.__new__(GeminiOCRBackend)
    backend.batch_size = 5
    backend.rate_limiter = RateLimiter(rpm_limit=10, tpm_limit=10_000, safety_margin=1.0)
    calls: list[int] = []

    async def fake_transcribe(batch: list[bytes]):
        calls.append(len(batch))
        return [("ok", None) for _ in batch]

    monkeypatch.setattr(backend, "_transcribe_batch", fake_transcribe)

    image = Image.new("RGB", (1024, 1024), "white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    results = await backend.transcribe_batch([buffer.getvalue()] * 3)

    # 1024x1024 costs 4,128 estimated tokens: two fit, three do not.
    assert calls == [2, 1]
    assert results == [("ok", None)] * 3


@pytest.mark.asyncio
async def test_slow_gemini_transport_does_not_block_the_event_loop():
    entered = Event()
    release = Event()

    class FakeModels:
        def generate_content(self, **kwargs):
            entered.set()
            assert release.wait(timeout=1)
            return SimpleNamespace(text="simulated Gemini reply")

    backend = object.__new__(GeminiOCRBackend)
    backend.rate_limiter = RateLimiter(rpm_limit=10, tpm_limit=100_000, safety_margin=1.0)
    backend._request_executor = ThreadPoolExecutor(max_workers=1)
    backend.client = SimpleNamespace(models=FakeModels())
    backend.max_output_tokens = 100
    backend.request_count = 0
    backend.timeout_seconds = 30
    image = Image.new("RGB", (1, 1), "white")
    image_buffer = BytesIO()
    image.save(image_buffer, format="JPEG")

    try:
        request = asyncio.create_task(
            backend._transcribe_with_model("simulation", [image_buffer.getvalue()], 1)
        )
        await asyncio.to_thread(entered.wait, 1)

        # This runs before the blocking SDK call is released.
        assert await asyncio.sleep(0, result="server still responsive") == "server still responsive"
        release.set()
        assert await request == ["simulated Gemini reply"]
    finally:
        backend._request_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_stuck_gemini_request_times_out_and_replaces_its_worker_pool():
    release = Event()

    class FakeModels:
        def generate_content(self, **kwargs):
            assert release.wait(timeout=1)
            return SimpleNamespace(text="too late")

    backend = object.__new__(GeminiOCRBackend)
    backend.rate_limiter = RateLimiter(rpm_limit=10, tpm_limit=100_000, safety_margin=1.0)
    old_executor = ThreadPoolExecutor(max_workers=1)
    backend._request_executor = old_executor
    backend.client = SimpleNamespace(models=FakeModels())
    backend.max_output_tokens = 100
    backend.request_count = 0
    backend.timeout_seconds = 0.05
    image = Image.new("RGB", (1, 1), "white")
    image_buffer = BytesIO()
    image.save(image_buffer, format="JPEG")

    try:
        with pytest.raises(GeminiRequestTimeout, match="exceeded"):
            await backend._transcribe_with_model("simulation", [image_buffer.getvalue()], 1)
        assert backend._request_executor is not old_executor
    finally:
        release.set()
        old_executor.shutdown(wait=True)
        backend._request_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_all_gemini_models_are_tried_before_local_fallback_can_run(monkeypatch):
    backend = object.__new__(GeminiOCRBackend)
    backend.primary_model = "primary"
    backend.batch_size = 1
    attempted_models: list[str] = []

    async def always_fail(model_name: str, images: list[bytes], estimated_tokens: int):
        attempted_models.append(model_name)
        raise GeminiRequestTimeout("simulated timeout")

    monkeypatch.setattr(backend, "_transcribe_with_model", always_fail)
    monkeypatch.setattr(backend, "_get_model_fallback_chain", lambda: ["primary", "fallback-a"])

    image = Image.new("RGB", (1, 1), "white")
    image_buffer = BytesIO()
    image.save(image_buffer, format="JPEG")

    with pytest.raises(RuntimeError, match="All Gemini models failed"):
        await backend._transcribe_batch([image_buffer.getvalue()])
    assert attempted_models == ["primary", "fallback-a"]
