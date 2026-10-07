import base64
import logging
from io import BytesIO
from pathlib import PurePath
from time import perf_counter

import fitz
import httpx
from charset_normalizer import from_bytes
from PIL import Image, UnidentifiedImageError

from app.config import settings

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".html", ".xml", ".yaml", ".yml"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
SYSTEM_PROMPT = (
    "Transcribe all visible text in the supplied document image exactly. "
    "Preserve the original language, Arabic script, punctuation, and line breaks as closely as possible. "
    "Do not translate, summarize, explain, or add text that is not visible. Return only the transcription."
)


async def _ask_ollama(client: httpx.AsyncClient, image_bytes: bytes) -> str:
    response = await client.post(
        f"{settings.ollama_base_url}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Transcribe the text in this image.",
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                },
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": settings.ollama_num_predict},
        },
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


async def _ask_llamacpp(client: httpx.AsyncClient, image_bytes: bytes) -> str:
    """Send an image to a llama.cpp server via its OpenAI-compatible /v1/chat/completions endpoint."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    response = await client.post(
        f"{settings.llamacpp_base_url}/chat/completions",
        json={
            "model": settings.llamacpp_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe the text in this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": settings.ollama_num_predict,
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _to_jpeg(image_bytes: bytes, quality: int = 85) -> bytes:
    """Re-encode image bytes as JPEG. Handles PNG pixmaps and arbitrary image formats."""
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()


async def _ask_model(image_bytes: bytes, timeout_seconds: float) -> str:
    """Route the request to the configured backend. Assumes image_bytes is JPEG. Creates its own client."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
        if settings.backend == "llamacpp":
            return await _ask_llamacpp(client, image_bytes)
        return await _ask_ollama(client, image_bytes)


async def _transcribe_image(image_bytes: bytes, timeout_seconds: float) -> str:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            buf = BytesIO()
            image.save(buf, format="JPEG", quality=85, optimize=True)
            jpeg_bytes = buf.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a valid, safe-to-process image.") from exc

    return await _ask_model(jpeg_bytes, timeout_seconds)


async def extract_attachment(filename: str, data: bytes) -> tuple[str, int | None]:
    """Return extracted text and the number of pages processed, when applicable."""
    started_at = perf_counter()
    suffix = PurePath(filename).suffix.lower()
    if settings.ocr_debug:
        logger.info("OCR started: file=%s type=%s bytes=%d model=%s", filename, suffix, len(data), settings.active_model)

    if suffix in TEXT_EXTENSIONS:
        decode_started_at = perf_counter()
        match = from_bytes(data).best()
        if match is None:
            raise ValueError("Could not determine the text file encoding.")
        text = str(match)
        if settings.ocr_debug:
            logger.info("Text decoding completed: file=%s duration=%.2fs", filename, perf_counter() - decode_started_at)
            logger.info("OCR finished: file=%s chars=%d total=%.2fs", filename, len(text), perf_counter() - started_at)
        return text, None

    if suffix == ".pdf":
        try:
            document = fitz.open(stream=data, filetype="pdf")
        except (fitz.FileDataError, ValueError) as exc:
            raise ValueError("The uploaded file is not a valid PDF.") from exc

        with document:
            if document.needs_pass:
                raise ValueError("Password-protected PDFs are not supported.")
            if len(document) == 0:
                raise ValueError("The PDF has no pages.")
            if len(document) > settings.max_pdf_pages:
                raise ValueError(f"PDF exceeds the {settings.max_pdf_pages}-page limit.")

            if settings.ocr_debug:
                logger.info("PDF opened: file=%s pages=%d", filename, len(document))
            pages = []
            for page_number, page in enumerate(document, start=1):
                page_started_at = perf_counter()
                render_started_at = perf_counter()
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                page_image = pixmap.tobytes("jpeg", jpg_quality=85)
                pixmap = None  # Release C-allocated pixmap memory immediately
                render_duration = perf_counter() - render_started_at
                if settings.ocr_debug:
                    logger.info(
                        "PDF page rendered: file=%s page=%d/%d duration=%.2fs image_bytes=%d",
                        filename,
                        page_number,
                        len(document),
                        render_duration,
                        len(page_image),
                    )
                inference_started_at = perf_counter()
                page_text = await _ask_model(page_image, settings.ollama_timeout_seconds)
                page_image = None  # Release JPEG bytes immediately after sending
                inference_duration = perf_counter() - inference_started_at
                pages.append(page_text)
                if settings.ocr_debug:
                    logger.info(
                        "PDF page completed: file=%s page=%d/%d model=%.2fs page_total=%.2fs chars=%d",
                        filename,
                        page_number,
                        len(document),
                        inference_duration,
                        perf_counter() - page_started_at,
                        len(page_text),
                    )
            text = "\n\n".join(pages).strip()
            if settings.ocr_debug:
                logger.info(
                    "OCR finished: file=%s pages=%d chars=%d total=%.2fs",
                    filename,
                    len(pages),
                    len(text),
                    perf_counter() - started_at,
                )
            return text, len(pages)

    if suffix in IMAGE_EXTENSIONS:
        page_started_at = perf_counter()
        text = await _transcribe_image(data, settings.ollama_timeout_seconds)
        if settings.ocr_debug:
            duration = perf_counter() - page_started_at
            logger.info("Image page completed: file=%s page=1/1 page_total=%.2fs chars=%d", filename, duration, len(text))
            logger.info("OCR finished: file=%s pages=1 chars=%d total=%.2fs", filename, len(text), perf_counter() - started_at)
        return text, 1

    raise ValueError(f"Unsupported file type: {suffix or '(no extension)'}.")


def image_from_data_url(url: str) -> bytes:
    """Decode an inline image supplied by OpenAI-compatible clients."""
    try:
        header, encoded = url.split(",", 1)
        if not header.startswith("data:image/") or ";base64" not in header:
            raise ValueError
        return base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("Images must be supplied as base64 data URLs.") from exc