# Gemini Empty Response - Troubleshooting

## Issue: Empty Response from Gemini

You're getting `WARNING: Gemini returned empty response for 20 images`

## Root Cause

**20 pages × 12,384 tokens/page = 247,756 tokens PER REQUEST**

This exceeds:
- ✗ Gemini's per-request context limit (likely ~128K-200K)
- ✗ Your 220K TPM limit wouldn't help (that's per MINUTE, not per request)

## The Fix

Reduce `GEMINI_BATCH_SIZE` to stay under per-request limits:

```env
# WRONG - Too many tokens per request
GEMINI_BATCH_SIZE=20  # 247K tokens = FAILS

# CORRECT - Stays under per-request limit  
GEMINI_BATCH_SIZE=5   # ~62K tokens = WORKS
GEMINI_BATCH_SIZE=10  # ~124K tokens = WORKS (closer to limit)
```

## Recommended Settings

```env
GEMINI_BATCH_SIZE=5
GEMINI_RPM_LIMIT=12
GEMINI_TPM_LIMIT=220000
GEMINI_SAFETY_MARGIN=0.9
```

### Why 5 Pages Works

- 5 pages × 12,384 tokens/page = **~62K tokens/request**
- Well under any reasonable per-request limit
- Still efficient: 50-page PDF = 10 requests (within 12 RPM)
- Uses ~620K tokens total (spread over 10 requests)

## TPM vs Per-Request Limits

These are DIFFERENT limits:

| Limit | What It Means | Your Value |
|-------|---------------|------------|
| **TPM** | Total tokens across ALL requests in 1 minute | 220K |
| **Per-Request** | Max tokens in a SINGLE request | ~128K-200K (Gemini internal) |

You can make multiple 60K-token requests per minute (up to 220K total), but each single request must stay under ~128K-200K.

## Performance Impact

With batch_size=5:
- 100-page PDF = 20 requests
- At 12 RPM limit: ~1.67 minutes total
- At 220K TPM limit: ~2.8 minutes total (bottleneck)
- **Throughput: ~35-60 pages/minute**

Still much better than batch_size=1 (100 requests = 8+ minutes)!

## Next Steps

1. **Set `GEMINI_BATCH_SIZE=5`** in `.env`
2. **Restart the service**
3. **Try your PDF again**

If it still fails, check for:
- Safety filters (content blocking)
- API key issues
- Model availability
