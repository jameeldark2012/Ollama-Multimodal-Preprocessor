# Checkpoint System

The OCR backend now includes an automatic checkpoint system that saves progress during PDF processing and enables resume from interruption.

## How It Works

1. **File Hashing**: Each uploaded PDF is hashed (SHA256, first 16 chars) to create a unique identifier
2. **Automatic Detection**: When you upload a PDF, the system checks if it's already been processed
3. **Three Scenarios**:
   - **New file**: Creates checkpoint, processes all pages
   - **Partial completion**: Resumes from last completed page
   - **Already completed**: Returns cached result **instantly** (no OCR needed!)

## Configuration

Add these to your `.env` file (already in `.env.example`):

```env
# Directory to store OCR checkpoints
CHECKPOINT_DIR=checkpoints

# Save checkpoint progress every N pages
CHECKPOINT_SAVE_INTERVAL=20
```

## Checkpoint Structure

```
checkpoints/
  {hash}-{timestamp}/
    metadata.json          # File info (hash, filename, total pages, created_at)
    progress.json          # Current state (completed_pages, total_pages, status)
    pages/
      page_0001.txt        # Extracted text for page 1
      page_0002.txt        # Extracted text for page 2
      ...
```

## Benefits

### For Your 517-Page Book Example:

**Without Checkpoints:**
- Process dies at page 342 → Start over from page 1
- Total waste: 342 pages of OCR work

**With Checkpoints:**
- Process dies at page 340 (last checkpoint at page 340)
- Re-upload same file → Resume from page 341
- Only process remaining 177 pages

**Best Case (Already Processed):**
- Upload the same PDF again → **Instant result** (reads from cache)
- No OCR processing needed at all!

## How to Use

**No changes needed!** The checkpoint system works automatically:

1. Upload PDF through Open WebUI's document loader (or `/v1/ocr` endpoint)
2. If processing times out or crashes, just re-upload the same file
3. System automatically detects the file (by hash) and resumes where it left off

## Checkpoint Persistence

**Checkpoints are NEVER automatically deleted.** They serve as a permanent cache:

- First upload of a PDF: Takes full processing time
- Subsequent uploads of the **same file**: Instant (returns cached text)
- Different PDF with same filename: Creates new checkpoint (different hash)

This means frequently-used PDFs (like reference documents) are processed once and cached forever.

## Testing

Run the test suite to verify checkpoint functionality:

```powershell
# Unit tests
.\.venv\Scripts\python.exe scripts\test_checkpoint.py

# Integration tests
.\.venv\Scripts\python.exe scripts\test_checkpoint_integration.py
```

## Debug Logging

Set `OCR_DEBUG=true` in `.env` to see checkpoint activity in logs:

```
PDF opened: file=book.pdf pages=517
Resuming from checkpoint: file=book.pdf hash=a3f5c8e2... completed=340/517
Skipping completed page: file=book.pdf page=1/517
...
Checkpoint saved: file=book.pdf progress=360/517
...
OCR finished: file=book.pdf pages=517 chars=985342 total=245.32s
```

## Manual Checkpoint Management

If you need to clear checkpoints (free up disk space):

```powershell
# Remove all checkpoints
Remove-Item -Recurse checkpoints

# Remove specific checkpoint by hash
Remove-Item -Recurse checkpoints\a3f5c8e2*
```

The system will recreate checkpoints as needed on next upload.
