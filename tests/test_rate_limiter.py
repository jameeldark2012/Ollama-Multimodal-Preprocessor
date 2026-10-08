"""
Comprehensive tests for RateLimiter to ensure it respects TPM and RPM limits.
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
    
    def test_estimate_text_tokens_minimum(self):
        """Empty text should return at least 1 token."""
        assert estimate_text_tokens("") == 1
        assert estimate_text_tokens("a") == 1
    
    def test_estimate_image_tokens(self):
        """Image token estimation based on 256x256 tiles."""
        # Single tile (256x256)
        tokens = estimate_image_tokens(256, 256)
        assert tokens == 258  # 1 tile * 258
        
        # 2x2 tiles (512x512)
        tokens = estimate_image_tokens(512, 512)
        assert tokens == 258 * 4  # 4 tiles
        
        # Typical PDF page (2000x1500)
        tokens = estimate_image_tokens(2000, 1500)
        tiles_w = (2000 + 255) // 256  # 8
        tiles_h = (1500 + 255) // 256  # 6
        expected = tiles_w * tiles_h * 258  # 8 * 6 * 258 = 12,384
        assert tokens == expected


class TestRateLimiterRPM:
    """Test RPM (requests per minute) limiting."""
    
    def test_rpm_basic(self):
        """Should allow requests up to RPM limit."""
        limiter = RateLimiter(rpm_limit=5, tpm_limit=100000, safety_margin=1.0)
        
        start = time.time()
        for i in range(5):
            limiter.wait(estimated_tokens=100)
        elapsed = time.time() - start
        
        # Should complete immediately (no waiting)
        assert elapsed < 0.5  # Allow 500ms for overhead
    
    def test_rpm_blocking(self):
        """Should block when RPM limit is exceeded."""
        limiter = RateLimiter(rpm_limit=3, tpm_limit=100000, safety_margin=1.0)
        
        start = time.time()
        
        # First 3 requests: instant
        for i in range(3):
            limiter.wait(estimated_tokens=100)
        
        # 4th request should wait ~60 seconds
        limiter.wait(estimated_tokens=100)
        
        elapsed = time.time() - start
        
        # Should have waited ~60 seconds (allow some margin)
        assert elapsed >= 55.0  # At least 55 seconds
        assert elapsed <= 65.0  # But not more than 65 seconds
    
    def test_rpm_safety_margin(self):
        """Safety margin should reduce effective RPM."""
        limiter = RateLimiter(rpm_limit=10, tpm_limit=100000, safety_margin=0.5)
        
        # Effective RPM = 10 * 0.5 = 5
        assert limiter.effective_rpm == 5
        
        start = time.time()
        
        # First 5 should be instant
        for i in range(5):
            limiter.wait(estimated_tokens=100)
        
        elapsed_first = time.time() - start
        assert elapsed_first < 0.5
        
        # 6th should block
        start_block = time.time()
        limiter.wait(estimated_tokens=100)
        elapsed_block = time.time() - start_block
        
        assert elapsed_block >= 55.0


class TestRateLimiterTPM:
    """Test TPM (tokens per minute) limiting."""
    
    def test_tpm_basic(self):
        """Should allow tokens up to TPM limit."""
        limiter = RateLimiter(rpm_limit=1000, tpm_limit=10000, safety_margin=1.0)
        
        start = time.time()
        # Use 9000 tokens total (under 10K limit)
        for i in range(3):
            limiter.wait(estimated_tokens=3000)
        elapsed = time.time() - start
        
        # Should complete immediately
        assert elapsed < 0.5
    
    def test_tpm_blocking(self):
        """Should block when TPM limit is exceeded."""
        limiter = RateLimiter(rpm_limit=1000, tpm_limit=5000, safety_margin=1.0)
        
        start = time.time()
        
        # First request: 3000 tokens (ok)
        limiter.wait(estimated_tokens=3000)
        
        # Second request: 3000 tokens (total 6000, exceeds 5000)
        # Should wait ~60 seconds for first request to expire
        limiter.wait(estimated_tokens=3000)
        
        elapsed = time.time() - start
        
        # Should have waited ~60 seconds
        assert elapsed >= 55.0
        assert elapsed <= 65.0
    
    def test_tpm_safety_margin(self):
        """Safety margin should reduce effective TPM."""
        limiter = RateLimiter(rpm_limit=1000, tpm_limit=10000, safety_margin=0.8)
        
        # Effective TPM = 10000 * 0.8 = 8000
        assert limiter.effective_tpm == 8000
        
        start = time.time()
        
        # 7000 tokens should be ok
        limiter.wait(estimated_tokens=7000)
        elapsed_first = time.time() - start
        assert elapsed_first < 0.5
        
        # Another 2000 tokens (total 9000) should block
        start_block = time.time()
        limiter.wait(estimated_tokens=2000)
        elapsed_block = time.time() - start_block
        
        assert elapsed_block >= 55.0


class TestRateLimiterIntegration:
    """Test combined RPM and TPM limiting (realistic scenarios)."""
    
    def test_gemini_free_tier_simulation(self):
        """Simulate Gemini free tier: 10 RPM, 65K TPM."""
        limiter = RateLimiter(rpm_limit=10, tpm_limit=65000, safety_margin=0.8)
        
        # Effective limits: 8 RPM, 52K TPM
        assert limiter.effective_rpm == 8
        assert limiter.effective_tpm == 52000
        
        start = time.time()
        
        # 5 requests with 5K tokens each (25K total)
        for i in range(5):
            limiter.wait(estimated_tokens=5000)
        
        elapsed = time.time() - start
        # Should be immediate (under both limits)
        assert elapsed < 1.0
    
    def test_gemini_paid_tier_simulation(self):
        """Simulate Gemini paid tier: 12 RPM, 220K TPM."""
        limiter = RateLimiter(rpm_limit=12, tpm_limit=220000, safety_margin=0.9)
        
        # Effective limits: 10.8 RPM, 198K TPM
        assert limiter.effective_rpm == 10  # Rounded down
        assert limiter.effective_tpm == 198000
        
        start = time.time()
        
        # 8 requests with 20K tokens each (160K total)
        for i in range(8):
            limiter.wait(estimated_tokens=20000)
        
        elapsed = time.time() - start
        # Should be immediate (under both limits)
        assert elapsed < 1.0
    
    def test_batch_processing_scenario(self):
        """Test realistic batch processing with 20 pages."""
        limiter = RateLimiter(rpm_limit=12, tpm_limit=220000, safety_margin=0.9)
        
        # Each batch: 20 pages × 3000 tokens = 60K tokens
        # Effective TPM: 198K
        # Can fit: 198K / 60K = 3.3 batches per minute
        
        start = time.time()
        
        # First 3 batches should be immediate
        for i in range(3):
            limiter.wait(estimated_tokens=60000)
        
        elapsed_first = time.time() - start
        assert elapsed_first < 1.0  # 180K tokens, under 198K limit
        
        # 4th batch (total 240K) should wait
        start_wait = time.time()
        limiter.wait(estimated_tokens=60000)
        elapsed_wait = time.time() - start_wait
        
        # Should wait for oldest batch to expire (~60s)
        assert elapsed_wait >= 55.0
    
    def test_sliding_window(self):
        """Test that sliding 60-second window works correctly."""
        limiter = RateLimiter(rpm_limit=5, tpm_limit=100000, safety_margin=1.0)
        
        # Make 3 requests immediately
        for i in range(3):
            limiter.wait(estimated_tokens=1000)
        
        # Wait 61 seconds (requests should expire)
        time.sleep(61)
        
        # Should be able to make 5 more requests immediately
        start = time.time()
        for i in range(5):
            limiter.wait(estimated_tokens=1000)
        elapsed = time.time() - start
        
        assert elapsed < 1.0  # Should be immediate


class TestRateLimiterEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_tokens(self):
        """Zero tokens should not block."""
        limiter = RateLimiter(rpm_limit=5, tpm_limit=10000, safety_margin=1.0)
        
        start = time.time()
        limiter.wait(estimated_tokens=0)
        elapsed = time.time() - start
        
        assert elapsed < 0.1
    
    def test_minimum_limits(self):
        """Limiter should enforce minimum values."""
        limiter = RateLimiter(rpm_limit=0, tpm_limit=-100, safety_margin=0.0)
        
        # Should be clamped to minimums
        assert limiter.rpm_limit >= 1
        assert limiter.tpm_limit >= 1
        assert limiter.safety_margin >= 0.05
    
    def test_maximum_safety_margin(self):
        """Safety margin should be capped at 1.0."""
        limiter = RateLimiter(rpm_limit=10, tpm_limit=10000, safety_margin=2.0)
        
        assert limiter.safety_margin == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
