"""
Quick tests for RateLimiter (no blocking tests).
Run these first to verify basic functionality without waiting.
"""
import time
import pytest
from app.rate_limiter import RateLimiter, estimate_text_tokens, estimate_image_tokens


class TestTokenEstimation:
    """Test token estimation functions."""
    
    def test_estimate_text_tokens(self):
        """Text token estimation should be ~1 token per 4 chars."""
        text = "a" * 400  # 400 characters
        tokens = estimate_text_tokens(text)
        assert tokens == 100  # 400 / 4
        print(f"✓ Text tokens: 400 chars = {tokens} tokens")
    
    def test_estimate_image_tokens(self):
        """Image token estimation based on 256x256 tiles."""
        # Typical PDF page (2000x1500)
        tokens = estimate_image_tokens(2000, 1500)
        tiles_w = (2000 + 255) // 256  # 8
        tiles_h = (1500 + 255) // 256  # 6
        expected = tiles_w * tiles_h * 258  # 8 * 6 * 258 = 12,384
        assert tokens == expected
        print(f"✓ Image tokens: 2000×1500px = {tokens} tokens ({tiles_w}×{tiles_h} tiles)")


class TestRateLimiterBasics:
    """Test basic rate limiter functionality (no blocking)."""
    
    def test_initialization(self):
        """Test limiter initialization with your settings."""
        limiter = RateLimiter(rpm_limit=12, tpm_limit=220000, safety_margin=0.9)
        
        assert limiter.rpm_limit == 12
        assert limiter.tpm_limit == 220000
        assert limiter.safety_margin == 0.9
        assert limiter.effective_rpm == 10  # 12 * 0.9 = 10.8 → 10
        assert limiter.effective_tpm == 198000  # 220000 * 0.9
        
        print(f"✓ Limiter initialized: {limiter.effective_rpm} RPM, {limiter.effective_tpm} TPM")
    
    def test_rpm_allows_under_limit(self):
        """Should allow requests under RPM limit without blocking."""
        limiter = RateLimiter(rpm_limit=12, tpm_limit=220000, safety_margin=0.9)
        
        start = time.time()
        # Make 5 requests (under 10.8 effective limit)
        for i in range(5):
            limiter.wait(estimated_tokens=1000)
        elapsed = time.time() - start
        
        assert elapsed < 0.5  # Should be instant
        print(f"✓ 5 requests completed in {elapsed:.3f}s (instant)")
    
    def test_tpm_allows_under_limit(self):
        """Should allow tokens under TPM limit without blocking."""
        limiter = RateLimiter(rpm_limit=12, tpm_limit=220000, safety_margin=0.9)
        
        start = time.time()
        # Use 150K tokens (under 198K effective limit)
        limiter.wait(estimated_tokens=150000)
        elapsed = time.time() - start
        
        assert elapsed < 0.5
        print(f"✓ 150K tokens allowed instantly (under 198K limit)")
    
    def test_realistic_batch_scenario(self):
        """Test realistic scenario: 3 batches of 20 pages."""
        limiter = RateLimiter(rpm_limit=12, tpm_limit=220000, safety_margin=0.9)
        
        # Each batch: 20 pages × 3000 tokens = 60K tokens
        start = time.time()
        
        for batch_num in range(3):
            limiter.wait(estimated_tokens=60000)
            print(f"  Batch {batch_num + 1}: 60K tokens processed")
        
        elapsed = time.time() - start
        
        # 180K total tokens, under 198K limit - should be instant
        assert elapsed < 1.0
        print(f"✓ 3 batches (180K tokens) completed in {elapsed:.3f}s")
    
    def test_detects_would_exceed_tpm(self):
        """Verify limiter would block when TPM would be exceeded."""
        limiter = RateLimiter(rpm_limit=100, tpm_limit=10000, safety_margin=1.0)
        
        # Use 8000 tokens
        limiter.wait(estimated_tokens=8000)
        
        # Check internal state
        now = time.time()
        recent_tokens = sum(t for s, t in limiter._timestamps if now - s < 60)
        
        assert recent_tokens == 8000
        print(f"✓ Tracked usage: {recent_tokens} / 10000 tokens")
        
        # Another 5000 would exceed (total 13000 > 10000)
        # We can't test the wait without blocking, but we verified tracking works


class TestEdgeCases:
    """Test edge cases."""
    
    def test_zero_tokens(self):
        """Zero tokens should not block."""
        limiter = RateLimiter(rpm_limit=12, tpm_limit=220000, safety_margin=0.9)
        
        start = time.time()
        limiter.wait(estimated_tokens=0)
        elapsed = time.time() - start
        
        assert elapsed < 0.1
        print(f"✓ Zero tokens allowed instantly")
    
    def test_minimum_limits_enforced(self):
        """Limiter should enforce minimum values."""
        limiter = RateLimiter(rpm_limit=0, tpm_limit=-100, safety_margin=0.0)
        
        assert limiter.rpm_limit >= 1
        assert limiter.tpm_limit >= 1
        assert limiter.safety_margin >= 0.05
        print(f"✓ Minimum limits enforced: {limiter.rpm_limit} RPM, {limiter.tpm_limit} TPM")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
