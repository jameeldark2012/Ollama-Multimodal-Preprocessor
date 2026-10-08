from __future__ import annotations

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
        limiter.wait(estimated_tokens=6000)   # blocks if needed, then records the call
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
        self._timestamps: deque[tuple[float, int]] = deque()
        self._request_times: deque[float] = deque()

    def wait(self, estimated_tokens: int = 0) -> None:
        """Block until both TPM and RPM budgets allow the next request, then record it."""
        token_delay = self._wait_seconds_for_tokens(estimated_tokens)
        if token_delay > 0:
            time.sleep(token_delay)
        rpm_delay = self._wait_seconds_for_rpm()
        if rpm_delay > 0:
            time.sleep(rpm_delay)

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
