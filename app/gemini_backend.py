"""
Gemini OCR Backend with Rate Limiting and Fallback

Features:
- Batch processing of multiple pages in one request
- RPM and TPM rate limiting with safety margins
- Model fallback chain: gemini-3.5-flash-lite -> gemini-3.1-flash-lite -> gemini-2.5-flash-lite
- Exponential backoff retry for 503/transient errors on the same model
- Falls back to local pipeline (Ollama/llama.cpp) on total Gemini failure
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image

from app.rate_limiter import RateLimiter, estimate_image_tokens, estimate_text_tokens

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Prompt design notes
# ---------------------------------------------------------------------------
# We deliberately use "copy" / "OCR scan" framing rather than "transcribe"
# because Gemini's RECITATION safety filter triggers when it detects the
# response would reproduce text it has memorised (e.g. Hadith collections).
# Framing the task as digitising a physical document image, and instructing
# the model to read character-by-character what it *sees*, sidesteps that
# filter in most cases.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an OCR engine scanning physical document images. "
    "Copy the text that is visually present on each scanned page image, "
    "reading character-by-character exactly what you see. "
    "Preserve the original language (Arabic or other), script, punctuation, "
    "and line structure as they appear in the scan. "
    "Do not draw on memory, do not paraphrase, do not translate. "
    "Output only the characters you observe in the image."
)

# Gemini fallback models in priority order
GEMINI_MODEL_FALLBACK = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

# Transient error signals — retry same model with backoff on these
_TRANSIENT_PATTERNS = (
    "503",
    "unavailable",
    "overloaded",
    "resource exhausted",
    "rate limit",
    "quota",
    "try again",
)

# One short same-model retry for transient Gemini overloads. After that, move
# to the fallback model/local fallback rather than holding an OCR job hostage.
_RETRY_DELAYS = (5.0,)


class GeminiRequestTimeout(TimeoutError):
    """The synchronous Gemini SDK exceeded the configured per-request limit."""


class GeminiRecitationError(RuntimeError):
    """Gemini blocked the OCR response because of its RECITATION filter."""


def _is_transient(exc: Exception) -> bool:
    """Return True if the exception looks like a transient / server-side error."""
    msg = str(exc).lower()
    return any(p in msg for p in _TRANSIENT_PATTERNS)


def _is_recitation(exc: Exception) -> bool:
    """Return True if the exception was caused by a RECITATION finish reason."""
    return "recitation" in str(exc).lower()


class GeminiOCRBackend:
    """
    Google Gemini OCR backend with batch processing and rate limiting.

    Processes multiple pages in a single request when possible to maximise
    efficiency under free tier limits (10 RPM).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        batch_size: int = 5,
        rpm_limit: int = 10,
        tpm_limit: int = 65000,
        safety_margin: float = 0.8,
        timeout_seconds: float = 300,
        max_output_tokens: int = 8192,
    ) -> None:
        self.api_key = api_key
        self.primary_model = model
        self.batch_size = max(1, batch_size)
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        # Configure a transport-level timeout too. This is the mechanism that
        # interrupts a stuck HTTP read inside the synchronous Google SDK.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=max(1, int(timeout_seconds * 1000))),
        )

        # Rate limiter respects both RPM and TPM
        self.rate_limiter = RateLimiter(
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            safety_margin=safety_margin,
        )
        # Gemini's SDK is synchronous. Keep its long-lived network waits out
        # of asyncio's shared executor, which is also used for PDF rendering,
        # checkpoint I/O, and video work.
        self._request_executor = self._new_request_executor()

        self.request_count = 0

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Gemini backend initialised: model=%s batch_size=%d rpm=%d tpm=%d safety=%.1f%%",
                model,
                self.batch_size,
                rpm_limit,
                tpm_limit,
                safety_margin * 100,
            )

    def _new_request_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(
            max_workers=min(self.rate_limiter.effective_rpm, 16),
            thread_name_prefix="gemini-request",
        )

    def _replace_stuck_executor(self) -> None:
        """Quarantine workers whose synchronous SDK call ignored its timeout."""
        old_executor = self._request_executor
        self._request_executor = self._new_request_executor()
        old_executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transcribe_batch(self, images: list[bytes]) -> list[tuple[str, str | None]]:
        """Transcribe images while keeping each API request inside the TPM budget."""
        if not images:
            return []

        prompt_tokens = (
            estimate_text_tokens(SYSTEM_PROMPT)
            + estimate_text_tokens("Copy the text from these images.")
        )
        results: list[tuple[str, str | None]] = []
        batch: list[bytes] = []
        batch_tokens = prompt_tokens

        for image in images[: self.batch_size]:
            with Image.open(BytesIO(image)) as opened_image:
                image_tokens = estimate_image_tokens(*opened_image.size)

            # Split configured batches before they exceed the limiter's
            # effective TPM budget. A lone huge image still has to be sent as
            # one request; RateLimiter reserves the entire minute for it.
            if batch and batch_tokens + image_tokens > self.rate_limiter.effective_tpm:
                results.extend(await self._transcribe_batch(batch))
                batch = []
                batch_tokens = prompt_tokens

            batch.append(image)
            batch_tokens += image_tokens

        if batch:
            results.extend(await self._transcribe_batch(batch))
        return results

    async def _transcribe_batch(self, images: list[bytes]) -> list[tuple[str, str | None]]:
        """
        Transcribe a batch of images (up to batch_size at once).

        Falls back through model list on failure.  Transient errors are
        retried on the same model with exponential back-off before trying
        the next model. RECITATION errors are raised immediately so the
        caller can use the local backend without spending more Gemini requests.

        Args:
            images: List of JPEG image bytes

        Returns:
            List of (text, failure_reason) tuples for each image.
            failure_reason is None for success, or one of:
            - "recitation": RECITATION safety filter blocked the response
            - "empty": Model returned empty text after all retries
            - "error": Fatal error (all models failed)

        Raises:
            RuntimeError: If all models (and per-page fallback) fail for entire batch
        """
        if not images:
            return []

        # Cap batch to configured size
        batch = images[: self.batch_size]

        # Estimate tokens for rate limiting
        total_tokens = (
            estimate_text_tokens(SYSTEM_PROMPT)
            + estimate_text_tokens("Copy the text from these images.")
        )
        for img_bytes in batch:
            with Image.open(BytesIO(img_bytes)) as img:
                width, height = img.size
                total_tokens += estimate_image_tokens(width, height)

        last_error: Exception | None = None
        for model_name in self._get_model_fallback_chain():
            # --- retry loop for transient errors on the same model ---
            for attempt in range(len(_RETRY_DELAYS) + 1):
                try:
                    texts = await self._transcribe_with_model(model_name, batch, total_tokens)
                    # Success! Return with no failure reasons
                    return [(text, None) for text in texts]
                except Exception as exc:
                    last_error = exc

                    if _is_recitation(exc):
                        logger.warning(
                            "Gemini model %s hit RECITATION for batch of %d images; "
                            "stopping Gemini attempts so the caller can use local fallback: %s",
                            model_name,
                            len(batch),
                            exc,
                        )
                        raise GeminiRecitationError(
                            f"Gemini RECITATION filter blocked batch of {len(batch)} images"
                        ) from exc

                    if _is_transient(exc) and attempt < len(_RETRY_DELAYS):
                        delay = _RETRY_DELAYS[attempt]
                        logger.warning(
                            "Gemini model %s transient error (attempt %d/%d), "
                            "retrying in %.0fs: %s",
                            model_name,
                            attempt + 1,
                            len(_RETRY_DELAYS) + 1,
                            delay,
                            exc,
                        )
                        await asyncio.sleep(delay)
                        continue  # retry same model

                    # Non-transient, non-RECITATION error → try next model
                    logger.warning(
                        "Gemini model %s failed for batch of %d images: %s",
                        model_name,
                        len(batch),
                        exc,
                    )
                    break

        # All Gemini models failed. Raise so OCR can invoke the configured
        # local fallback only after the complete Gemini chain is exhausted.
        raise RuntimeError(
            f"All Gemini models failed for batch of {len(batch)} images. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model_fallback_chain(self) -> list[str]:
        """Return model names in fallback priority order."""
        chain = [self.primary_model]
        for model in GEMINI_MODEL_FALLBACK:
            if model != self.primary_model:
                chain.append(model)
        return chain

    async def _transcribe_with_model(
        self, model_name: str, images: list[bytes], estimated_tokens: int
    ) -> list[str]:
        """
        Transcribe images using a specific model.

        Args:
            model_name: Gemini model to use
            images: List of JPEG image bytes
            estimated_tokens: Pre-calculated token estimate for rate limiting

        Returns:
            List of transcribed text for each image

        Raises:
            Exception: On API or parsing error
        """
        # Wait for rate limits
        await self.rate_limiter.wait(estimated_tokens=estimated_tokens)
        self.request_count += 1

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Gemini API request #%d: model=%s batch_size=%d est_tokens=%d",
                self.request_count,
                model_name,
                len(images),
                estimated_tokens,
            )

        # Build prompt
        if len(images) == 1:
            img = Image.open(BytesIO(images[0]))
            contents = [
                SYSTEM_PROMPT,
                img,
                "Copy all text visible in this scanned document image.",
            ]
        else:
            # Use a marker format that is visually distinctive and very unlikely
            # to appear in manuscript text: <<<PAGE_N>>>
            contents = [SYSTEM_PROMPT]
            for i, img_bytes in enumerate(images, start=1):
                img = Image.open(BytesIO(img_bytes))
                contents.append(f"[SCAN {i} OF {len(images)}]")
                contents.append(img)

            marker_examples = "\n".join(
                f"<<<PAGE_{i}>>>\n<text copied from scan {i}>" for i in range(1, len(images) + 1)
            )
            contents.append(
                f"Copy all text from each of the {len(images)} scanned document images above. "
                f"Read character-by-character what you see in each scan — do not draw on memory. "
                f"You MUST output exactly {len(images)} sections. "
                f"Begin each section with the exact marker <<<PAGE_N>>> on its own line "
                f"(where N is the scan number, starting at 1). "
                f"Do not add any commentary, headings, or text outside the markers. "
                f"Output token budget: {self.max_output_tokens}. "
                f"Required format (example for {len(images)} scans):\n"
                f"{marker_examples}"
            )

        # Call Gemini API - SDK is sync, so run in thread pool
        call = partial(
            self.client.models.generate_content,
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        future = asyncio.get_running_loop().run_in_executor(self._request_executor, call)
        try:
            # Defensive timeout: protects the API request even if a transport
            # bug causes the SDK's own HTTP timeout to be ignored.
            response = await asyncio.wait_for(future, timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            self._replace_stuck_executor()
            raise GeminiRequestTimeout(
                f"Gemini request exceeded {self.timeout_seconds:.0f}s timeout"
            ) from exc

        # Extract text
        text: str | None = None
        if hasattr(response, "text") and response.text:
            text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                if hasattr(candidate.content, "parts") and candidate.content.parts:
                    text = candidate.content.parts[0].text

        # Check for blocks / empty response
        if not text:
            error_details: list[str] = []
            if hasattr(response, "prompt_feedback"):
                error_details.append(f"prompt_feedback={response.prompt_feedback}")
            finish_reason = None
            safety_ratings = None
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason"):
                    finish_reason = candidate.finish_reason
                    error_details.append(f"finish_reason={finish_reason}")
                if hasattr(candidate, "safety_ratings"):
                    safety_ratings = candidate.safety_ratings
                    error_details.append(f"safety_ratings={safety_ratings}")

            error_msg = f"Gemini returned empty response for {len(images)} images"
            if error_details:
                error_msg += f" ({', '.join(error_details)})"
            raise RuntimeError(error_msg)

        # Parse response
        if len(images) == 1:
            return [text.strip()]
        return self._parse_batch_response(text, len(images))

    # ------------------------------------------------------------------
    # Batch response parser
    # ------------------------------------------------------------------

    def _parse_batch_response(self, text: str, expected_pages: int) -> list[str]:
        """
        Parse a batched response that should contain *expected_pages* sections.

        Strategies tried in order:
        1. New <<<PAGE_N>>> markers (exact count)
        2. Legacy ===PAGE_START_N markers (exact count)
        3. Dash-style --- Page N --- markers (exact count)
        4. Any marker variant with matching count
        5. Accept N-1 from any strategy above (last page missing / cut off)
        6. Triple-newline split (exact)
        7. Double-newline split — pick best N consecutive segments
        """
        # --- Strategy 1: new <<<PAGE_N>>> markers ---
        new_marker = re.compile(r"<<<PAGE_(\d+)>>>", re.IGNORECASE)
        new_matches = list(new_marker.finditer(text))

        # --- Strategy 2: legacy ===PAGE_START_N markers ---
        strict_pattern = re.compile(r"===PAGE_START_(\d+)", re.IGNORECASE)
        strict_matches = list(strict_pattern.finditer(text))

        # --- Strategy 3: dash markers --- Page N --- ---
        dash_pattern = re.compile(r"-{2,3}\s*[Pp]age\s+(\d+)\s*-{2,3}")
        dash_matches = list(dash_pattern.finditer(text))

        # --- Strategy 4: any consistent marker with matching count ---
        any_marker = re.compile(
            r"(?:={2,}|[-]{2,}|<{2,3})\s*(?:PAGE|Page|page)\s*[_\-]?\s*"
            r"(?:START\s*[_\-]?\s*)?(\d+)\s*(?:={2,}|[-]{2,}|>{2,3})?"
        )
        any_matches = list(any_marker.finditer(text))

        def _extract_segments(matches: list) -> list[str]:
            pages = []
            for idx, match in enumerate(matches):
                start = match.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                pages.append(text[start:end].strip())
            return pages

        # Check each strategy for exact match first, then N-1 fallback
        candidate_strategies: list[tuple[str, list]] = [
            ("new_marker", new_matches),
            ("strict", strict_matches),
            ("dash", dash_matches),
            ("any", any_matches),
        ]

        for strategy_name, matches in candidate_strategies:
            if len(matches) == expected_pages:
                return _extract_segments(matches)

        # N-1 fallback: last page was cut off or had no content — pad with ""
        for strategy_name, matches in candidate_strategies:
            if len(matches) == expected_pages - 1 and len(matches) > 0:
                segments = _extract_segments(matches)
                logger.warning(
                    "Batch parser [%s] found %d/%d markers — last page missing, "
                    "appending empty page",
                    strategy_name,
                    len(matches),
                    expected_pages,
                )
                segments.append("")
                return segments

        # --- Strategy 5: triple-newline splitting ---
        split_pages = [p.strip() for p in re.split(r"\n{3,}", text) if p.strip()]
        if len(split_pages) == expected_pages:
            return split_pages

        # --- Strategy 6: double-newline splitting ---
        # Pick the best N *consecutive* segments (the ones with the most content)
        split_pages2 = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(split_pages2) >= expected_pages:
            if len(split_pages2) == expected_pages:
                best = split_pages2
            else:
                # Find the window of N consecutive segments with the most total characters
                best_start = 0
                best_len = sum(len(s) for s in split_pages2[:expected_pages])
                for start in range(1, len(split_pages2) - expected_pages + 1):
                    window_len = sum(len(s) for s in split_pages2[start : start + expected_pages])
                    if window_len > best_len:
                        best_len = window_len
                        best_start = start
                best = split_pages2[best_start : best_start + expected_pages]

            logger.warning(
                "Batch parser used double-newline fallback: got %d segments for %d pages",
                len(split_pages2),
                expected_pages,
            )
            return best

        # --- Nothing worked ---
        logger.warning(
            "Batch parser could not split response into %d pages. "
            "Strategies found: new_marker=%d strict=%d dash=%d any=%d "
            "triple_nl=%d double_nl=%d. "
            "Response preview: %.200s",
            expected_pages,
            len(new_matches),
            len(strict_matches),
            len(dash_matches),
            len(any_matches),
            len(split_pages),
            len(split_pages2),
            text[:200],
        )
        raise RuntimeError(
            f"Gemini returned only {len(split_pages2) or 1} transcriptions "
            f"for {expected_pages} pages. "
            "Response may be incomplete or incorrectly formatted."
        )
