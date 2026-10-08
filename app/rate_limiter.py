from __future__ import annotations

import asyncio
import time
from collections import deque

DEFAULT_RPM_LIMIT = 10
DEFAULT_TPM_LIMIT = 65000
DEFAULT_SAFETY_MARGIN = 0.8


def estimate_text_tokens(text: str) -> int:
    """Estimate text token count for rate limiting and debug logging."""
    return max(1, len(text) // 4)


def estimate_image_tokens(width: int, height: int) -> int:
    """
    Estimate Gemini image token cost.
    Gemini charges ~258 tokens per 256x256 tile.
    """
    tiles_w = (width + 255) // 256
    tiles_h = (height + 255) // 256
    return tiles_w * tiles_h * 258


class RateLimiter:
    """
    Token-per-minute and request-per-minute rate limiter for AI API calls.

    Usage:
        limiter = RateLimiter(tpm_limit=65000, rpm_limit=10, safety_margin=0.8)
        await limiter.wait(estimated_tokens=6000)   # async waits if needed, then records the call
    """

    def __init__(
        self,
        *,
        tpm_limit: int = DEFAULT_TPM_LIMIT,
        rpm_limit: int = DEFAULT_RPM_LIMIT,
        safety_margin: float = DEFAULT_SAFETY_MARGIN,
    ) -> None:
        self.tpm_limit = max(1, int(tpm_limit))
        self.rpm_limit = max(1, int(rpm_limit))
        self.safety_margin = max(0.05, min(1.0, float(safety_margin)))
        self.effective_tpm = max(1, int(self.tpm_limit * self.safety_margin))
        self.effective_rpm = max(1, int(self.rpm_limit * self.safety_margin))
        # Entries are reservations for request start times.  Future entries are
        # intentional: they keep concurrent coroutines from selecting the same
        # available slot while one of them is asleep.
        self._reservations: deque[tuple[float, int]] = deque()
        # Kept as compatibility aliases for diagnostics and existing callers.
        self._timestamps = self._reservations
        self._request_times: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def wait(self, estimated_tokens: int = 0) -> None:
        """Reserve and wait for one request slot without blocking the event loop."""
        # A single image can legitimately estimate above a conservative TPM
        # budget.  It cannot be split further here, so reserve a full minute's
        # budget rather than repeatedly scheduling it forever.
        tokens = min(max(0, estimated_tokens), self.effective_tpm)
        async with self._lock:
            now = time.monotonic()
            while self._reservations and self._reservations[0][0] <= now - 60:
                self._reservations.popleft()

            scheduled_at = now
            while True:
                window = [(timestamp, cost) for timestamp, cost in self._reservations if timestamp > scheduled_at - 60]
                token_total = sum(cost for _, cost in window)
                request_total = len(window)
                if token_total + tokens <= self.effective_tpm and request_total < self.effective_rpm:
                    break

                expirations = [timestamp + 60 for timestamp, _ in window if timestamp + 60 > scheduled_at]
                # There is always an expiration when a non-empty window is
                # over either budget; this guard also prevents a busy loop.
                scheduled_at = min(expirations) if expirations else scheduled_at + 60

            self._reservations.append((scheduled_at, tokens))
            self._request_times.append(scheduled_at)
            delay = max(0.0, scheduled_at - now)

        if delay:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                # Do not burn a quota slot for a request that will never be
                # sent (for example, when its HTTP request is cancelled).
                async with self._lock:
                    try:
                        self._reservations.remove((scheduled_at, tokens))
                        self._request_times.remove(scheduled_at)
                    except ValueError:
                        pass
                raise

    def _wait_seconds_for_tokens(self, estimated_tokens: int) -> float:
        if estimated_tokens <= 0:
            return 0.0
        now = time.monotonic()
        recent = [(s, t) for s, t in self._timestamps if now - s < 60]
        total = sum(t for _, t in recent) + estimated_tokens
        if total <= self.effective_tpm:
            self._timestamps.append((now, estimated_tokens))
            return 0.0
        oldest = recent[0][0] if recent else now
        wait = max(0.0, 60.0 - max(0.1, now - oldest))
        self._timestamps.append((now + wait, estimated_tokens))
        return wait

    def _wait_seconds_for_rpm(self) -> float:
        now = time.monotonic()
        while self._request_times and now - self._request_times[0] >= 60:
            self._request_times.popleft()
            now = time.monotonic()
        if len(self._request_times) < self.effective_rpm:
            self._request_times.append(now)
            return 0.0
        oldest = self._request_times[0]
        wait = max(0.0, 60.0 - (now - oldest))
        self._request_times.append(now + wait if wait > 0 else now)
        return wait
