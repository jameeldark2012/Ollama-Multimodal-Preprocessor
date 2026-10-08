import base64
import logging
from datetime import datetime
from io import BytesIO
from pathlib import PurePath
from time import perf_counter

import fitz
import httpx
from charset_normalizer import from_bytes
from PIL import Image, UnidentifiedImageError

from app.checkpoint import CheckpointManager
from app.config import settings

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)
checkpoint_manager = CheckpointManager(settings.checkpoint_dir)

# Initialize Gemini backend if configured
gemini_backend = None
if settings.backend == "gemini" and settings.gemini_api_key:
    try:
        from app.gemini_backend import GeminiOCRBackend
        gemini_backend = GeminiOCRBackend(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            batch_size=settings.gemini_batch_size,
            rpm_limit=settings.gemini_rpm_limit,
            tpm_limit=settings.gemini_tpm_limit,
            safety_margin=settings.gemini_safety_margin,
            timeout_seconds=settings.gemini_timeout_seconds,
            max_output_tokens=settings.gemini_max_output_tokens,
        )
        logger.info("Gemini backend initialized successfully")
    except Exception as exc:
        logger.error("Failed to initialize Gemini backend: %s", exc)
        if not settings.gemini_fallback_to_local:
            raise

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


async def _ask_model(image_bytes: bytes, timeout_seconds: float, force_local: bool = False) -> str:
    """
    Route the request to the configured backend with fallback support.
    
    If Gemini is configured and fails, falls back to local (Ollama/llama.cpp) if enabled.
    Assumes image_bytes is JPEG. Creates its own client for HTTP backends.
    
    Args:
        image_bytes: JPEG image bytes
        timeout_seconds: Request timeout
        force_local: If True, skip Gemini and go straight to local backend
    """
    # Try Gemini first if configured AND not forcing local
    if not force_local and settings.backend == "gemini" and gemini_backend is not None:
        try:
            results = gemini_backend.transcribe_batch([image_bytes])
            return results[0] if results else ""
        except Exception as exc:
            logger.error("Gemini backend failed: %s", exc)
            if settings.gemini_fallback_to_local:
                logger.info("Falling back to local backend (Ollama/llama.cpp)")
            else:
                raise

    # Determine which local backend to use.
    # When falling back from Gemini, honour GEMINI_LOCAL_FALLBACK_BACKEND if set.
    local_backend = settings.backend
    if local_backend == "gemini":
        local_backend = settings.gemini_local_fallback_backend or "ollama"

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
        if local_backend == "llamacpp":
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

            total_pages = len(document)
            if settings.ocr_debug:
                logger.info("PDF opened: file=%s pages=%d", filename, total_pages)

            # Calculate file hash and check for existing checkpoint
            file_hash = checkpoint_manager.calculate_file_hash(data)
            checkpoint_info = checkpoint_manager.find_checkpoint(file_hash)

            if checkpoint_info:
                # Resume from existing checkpoint
                completed = checkpoint_info.completed_pages
                if checkpoint_info.status == "completed" and completed == total_pages:
                    # Already fully processed - return cached result immediately
                    if settings.ocr_debug:
                        logger.info(
                            "PDF already completed in checkpoint: file=%s hash=%s pages=%d",
                            filename,
                            file_hash,
                            total_pages,
                        )
                    pages = checkpoint_manager.load_all_pages(checkpoint_info.checkpoint_dir, total_pages)
                    non_empty_pages = [p for p in pages if p.strip()]
                    text = "\n\n".join(non_empty_pages).strip()
                    if settings.ocr_debug:
                        logger.info(
                            "OCR finished (cached): file=%s pages=%d (non-empty=%d) chars=%d total=%.2fs",
                            filename,
                            total_pages,
                            len(non_empty_pages),
                            len(text),
                            perf_counter() - started_at,
                        )
                    return text, len(non_empty_pages)
                else:
                    # Resume partial processing
                    if settings.ocr_debug:
                        logger.info(
                            "Resuming from checkpoint: file=%s hash=%s completed=%d/%d",
                            filename,
                            file_hash,
                            completed,
                            total_pages,
                        )
                    checkpoint_dir = checkpoint_info.checkpoint_dir
                    completed_page_numbers = checkpoint_manager.get_completed_page_numbers(checkpoint_dir)
                    pages = checkpoint_manager.load_all_pages(checkpoint_dir, total_pages)
            else:
                # Create new checkpoint
                checkpoint_dir = checkpoint_manager.create_checkpoint(file_hash, filename, total_pages)
                completed_page_numbers = set()
                pages = [""] * total_pages
                if settings.ocr_debug:
                    logger.info("Created new checkpoint: file=%s hash=%s pages=%d", filename, file_hash, total_pages)

            # Process pages - batch when using Gemini, one-by-one otherwise
            pages_processed = 0
            pending_pages: list[tuple[int, bytes]] = []  # (page_number, image_bytes)
            failed_pages_metadata: dict[int, dict] = checkpoint_manager.load_failed_pages(checkpoint_dir)
            
            # Determine batch size
            batch_size = gemini_backend.batch_size if (settings.backend == "gemini" and gemini_backend) else 1
            
            # Extract pages that need retry (not RECITATION, attempts < 2)
            pages_to_retry: list[tuple[int, bytes]] = []
            for page_num in range(1, total_pages + 1):
                if page_num in completed_page_numbers:
                    continue  # Already successfully processed
                if page_num in failed_pages_metadata:
                    meta = failed_pages_metadata[page_num]
                    if meta["reason"] != "recitation" and meta["attempts"] < 2:
                        # This page needs retry - render it
                        page_started_at = perf_counter()
                        page = document[page_num - 1]
                        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                        page_image = pixmap.tobytes("jpeg", jpg_quality=85)
                        pixmap = None
                        pages_to_retry.append((page_num, page_image))
                        if settings.ocr_debug:
                            logger.info(
                                "Queueing retry: file=%s page=%d/%d reason=%s attempts=%d",
                                filename,
                                page_num,
                                total_pages,
                                meta["reason"],
                                meta["attempts"],
                            )
            
            for page_number, page in enumerate(document, start=1):
                # Skip already completed pages
                if page_number in completed_page_numbers:
                    if settings.ocr_debug:
                        logger.info("Skipping completed page: file=%s page=%d/%d", filename, page_number, total_pages)
                    continue
                
                # Skip pages already queued for retry
                if any(pnum == page_number for pnum, _ in pages_to_retry):
                    continue

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
                        total_pages,
                        render_duration,
                        len(page_image),
                    )
                
                # Add to batch
                pending_pages.append((page_number, page_image))
                
                # Process batch when full or at end of document
                if len(pending_pages) >= batch_size or page_number == total_pages:
                    inference_started_at = perf_counter()
                    
                    if settings.backend == "gemini" and gemini_backend:
                        # Batch process with Gemini
                        try:
                            images = [img for _, img in pending_pages]
                            results = gemini_backend.transcribe_batch(images)
                            
                            # Process results with failure tracking
                            for (pnum, img), (text, reason) in zip(pending_pages, results):
                                if reason is None:
                                    # Success!
                                    pages[pnum - 1] = text
                                    pages_processed += 1
                                    checkpoint_manager.save_page(checkpoint_dir, pnum, text)
                                    # Remove from failed metadata if it was there
                                    if pnum in failed_pages_metadata:
                                        del failed_pages_metadata[pnum]
                                else:
                                    # Failure - track it
                                    if pnum in failed_pages_metadata:
                                        failed_pages_metadata[pnum]["attempts"] += 1
                                        failed_pages_metadata[pnum]["reason"] = reason
                                    else:
                                        failed_pages_metadata[pnum] = {
                                            "reason": reason,
                                            "attempts": 1,
                                            "last_tried": datetime.now().isoformat(),
                                        }
                                    
                                    # Only queue for retry if not RECITATION and attempts < 2
                                    if reason != "recitation" and failed_pages_metadata[pnum]["attempts"] < 2:
                                        pages_to_retry.append((pnum, img))
                                        if settings.ocr_debug:
                                            logger.info(
                                                "Page %d/%d failed (reason=%s, attempts=%d) — queuing for retry",
                                                pnum,
                                                total_pages,
                                                reason,
                                                failed_pages_metadata[pnum]["attempts"],
                                            )
                                    else:
                                        logger.warning(
                                            "Page %d/%d permanently failed (reason=%s, attempts=%d) — marking as empty",
                                            pnum,
                                            total_pages,
                                            reason,
                                            failed_pages_metadata[pnum]["attempts"],
                                        )
                                        checkpoint_manager.save_page(checkpoint_dir, pnum, "")
                            
                            # Save failed pages metadata
                            checkpoint_manager.save_failed_pages(checkpoint_dir, failed_pages_metadata)

                        except Exception as exc:
                            logger.error("Gemini batch processing failed: %s", exc)
                            # Fall back to local processing if enabled
                            if settings.gemini_fallback_to_local:
                                logger.info("Falling back to local processing for batch of %d pages", len(pending_pages))
                                for pnum, img in pending_pages:
                                    page_text = await _ask_model(img, settings.ollama_timeout_seconds, force_local=True)
                                    if not page_text.strip():
                                        # Track as failed
                                        if pnum in failed_pages_metadata:
                                            failed_pages_metadata[pnum]["attempts"] += 1
                                        else:
                                            failed_pages_metadata[pnum] = {
                                                "reason": "empty",
                                                "attempts": 1,
                                                "last_tried": datetime.now().isoformat(),
                                            }
                                        if failed_pages_metadata[pnum]["attempts"] < 2:
                                            pages_to_retry.append((pnum, img))
                                        else:
                                            checkpoint_manager.save_page(checkpoint_dir, pnum, "")
                                    else:
                                        pages[pnum - 1] = page_text
                                        pages_processed += 1
                                        checkpoint_manager.save_page(checkpoint_dir, pnum, page_text)
                                        if pnum in failed_pages_metadata:
                                            del failed_pages_metadata[pnum]
                                checkpoint_manager.save_failed_pages(checkpoint_dir, failed_pages_metadata)
                            else:
                                raise
                    else:
                        # Process one-by-one with local backend
                        for pnum, img in pending_pages:
                            page_text = await _ask_model(img, settings.ollama_timeout_seconds)
                            if not page_text.strip():
                                if pnum in failed_pages_metadata:
                                    failed_pages_metadata[pnum]["attempts"] += 1
                                else:
                                    failed_pages_metadata[pnum] = {
                                        "reason": "empty",
                                        "attempts": 1,
                                        "last_tried": datetime.now().isoformat(),
                                    }
                                if failed_pages_metadata[pnum]["attempts"] < 2:
                                    pages_to_retry.append((pnum, img))
                                else:
                                    checkpoint_manager.save_page(checkpoint_dir, pnum, "")
                            else:
                                pages[pnum - 1] = page_text
                                pages_processed += 1
                                checkpoint_manager.save_page(checkpoint_dir, pnum, page_text)
                                if pnum in failed_pages_metadata:
                                    del failed_pages_metadata[pnum]
                        checkpoint_manager.save_failed_pages(checkpoint_dir, failed_pages_metadata)
                    
                    inference_duration = perf_counter() - inference_started_at
                    
                    # Update progress at configured interval
                    if pages_processed % settings.checkpoint_save_interval == 0 or page_number == total_pages:
                        completed_count = len(completed_page_numbers) + pages_processed
                        checkpoint_manager.update_progress(checkpoint_dir, completed_count, total_pages, "processing")
                        if settings.ocr_debug:
                            logger.info(
                                "Checkpoint saved: file=%s progress=%d/%d",
                                filename,
                                completed_count,
                                total_pages,
                            )

                    if settings.ocr_debug:
                        pages_in_batch = len(pending_pages)
                        logger.info(
                            "PDF batch completed: file=%s pages=%d model=%.2fs batch_total=%.2fs",
                            filename,
                            pages_in_batch,
                            inference_duration,
                            perf_counter() - page_started_at,
                        )
                    
                    # Clear batch
                    pending_pages = []

            # --- Retry phase: process retryable failed pages in proper batches ---
            if pages_to_retry:
                logger.info(
                    "Retrying %d pages that failed with transient errors — processing in batches of %d",
                    len(pages_to_retry),
                    batch_size,
                )
                # Sort by page number to maintain sequence
                pages_to_retry.sort(key=lambda x: x[0])
                
                retry_batch: list[tuple[int, bytes]] = []
                for pnum, img in pages_to_retry:
                    retry_batch.append((pnum, img))
                    
                    if len(retry_batch) >= batch_size or (pnum, img) == pages_to_retry[-1]:
                        # Process retry batch
                        if settings.backend == "gemini" and gemini_backend:
                            try:
                                retry_images = [img for _, img in retry_batch]
                                retry_results = gemini_backend.transcribe_batch(retry_images)
                                
                                for (rnum, _), (text, reason) in zip(retry_batch, retry_results):
                                    if reason is None:
                                        pages[rnum - 1] = text
                                        pages_processed += 1
                                        checkpoint_manager.save_page(checkpoint_dir, rnum, text)
                                        if rnum in failed_pages_metadata:
                                            del failed_pages_metadata[rnum]
                                    else:
                                        failed_pages_metadata[rnum]["attempts"] += 1
                                        failed_pages_metadata[rnum]["reason"] = reason
                                        logger.warning(
                                            "Page %d/%d still failed after retry (reason=%s, attempts=%d) — marking as empty",
                                            rnum,
                                            total_pages,
                                            reason,
                                            failed_pages_metadata[rnum]["attempts"],
                                        )
                                        checkpoint_manager.save_page(checkpoint_dir, rnum, "")
                                
                                checkpoint_manager.save_failed_pages(checkpoint_dir, failed_pages_metadata)
                            except Exception as exc:
                                logger.error("Gemini retry batch failed: %s", exc)
                                if settings.gemini_fallback_to_local:
                                    for rnum, rimg in retry_batch:
                                        page_text = await _ask_model(rimg, settings.ollama_timeout_seconds, force_local=True)
                                        if not page_text.strip():
                                            logger.warning(
                                                "Page %d/%d still empty after local fallback — marking done, skipping",
                                                rnum, total_pages,
                                            )
                                            checkpoint_manager.save_page(checkpoint_dir, rnum, "")
                                        else:
                                            pages[rnum - 1] = page_text
                                            pages_processed += 1
                                            checkpoint_manager.save_page(checkpoint_dir, rnum, page_text)
                                            if rnum in failed_pages_metadata:
                                                del failed_pages_metadata[rnum]
                                    checkpoint_manager.save_failed_pages(checkpoint_dir, failed_pages_metadata)
                                else:
                                    # Mark all as empty
                                    for rnum, _ in retry_batch:
                                        checkpoint_manager.save_page(checkpoint_dir, rnum, "")
                        else:
                            # Local backend retry
                            for rnum, rimg in retry_batch:
                                page_text = await _ask_model(rimg, settings.ollama_timeout_seconds)
                                if not page_text.strip():
                                    logger.warning(
                                        "Page %d/%d still empty after retry — marking done, skipping",
                                        rnum, total_pages,
                                    )
                                    checkpoint_manager.save_page(checkpoint_dir, rnum, "")
                                else:
                                    pages[rnum - 1] = page_text
                                    pages_processed += 1
                                    checkpoint_manager.save_page(checkpoint_dir, rnum, page_text)
                                    if rnum in failed_pages_metadata:
                                        del failed_pages_metadata[rnum]
                            checkpoint_manager.save_failed_pages(checkpoint_dir, failed_pages_metadata)
                        
                        retry_batch = []

            # Mark as completed
            checkpoint_manager.update_progress(checkpoint_dir, total_pages, total_pages, "completed")
            non_empty_pages = [p for p in pages if p.strip()]
            text = "\n\n".join(non_empty_pages).strip()
            if settings.ocr_debug:
                logger.info(
                    "OCR finished: file=%s pages=%d (non-empty=%d) chars=%d total=%.2fs",
                    filename,
                    total_pages,
                    len(non_empty_pages),
                    len(text),
                    perf_counter() - started_at,
                )
            return text, len(non_empty_pages)

    if suffix in IMAGE_EXTENSIONS:
        page_started_at = perf_counter()
        text = await _transcribe_image(data, settings.ollama_timeout_seconds)
        if not text.strip():
            logger.warning("Model returned empty text for image: file=%s", filename)
        pages_out = 1 if text.strip() else 0
        if settings.ocr_debug:
            duration = perf_counter() - page_started_at
            logger.info("Image page completed: file=%s page=1/1 page_total=%.2fs chars=%d", filename, duration, len(text))
            logger.info("OCR finished: file=%s pages=%d chars=%d total=%.2fs", filename, pages_out, len(text), perf_counter() - started_at)
        return text, pages_out

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