"""Checkpoint manager for OCR processing with file-based resume capability."""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import aiofiles

logger = logging.getLogger("uvicorn.error")


class CheckpointInfo(NamedTuple):
    """Information about an existing checkpoint."""
    checkpoint_dir: Path
    file_hash: str
    completed_pages: int
    total_pages: int
    status: str  # "processing", "completed", "failed"


class CheckpointManager:
    """Manages file-based checkpoints for OCR processing."""

    def __init__(self, base_dir: str):
        """Initialize checkpoint manager with base directory."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def calculate_file_hash(self, data: bytes) -> str:
        """Calculate SHA256 hash of file content (first 16 hex chars) - async."""
        return await asyncio.to_thread(lambda: hashlib.sha256(data).hexdigest()[:16])

    async def find_checkpoint(self, file_hash: str) -> CheckpointInfo | None:
        """Find existing checkpoint by file hash. Returns most recent if multiple exist."""
        # glob is sync, run in thread
        matching_dirs = await asyncio.to_thread(
            lambda: list(self.base_dir.glob(f"{file_hash}-*"))
        )
        if not matching_dirs:
            return None

        # Sort by timestamp (newest first) and take the most recent
        matching_dirs.sort(reverse=True)
        checkpoint_dir = matching_dirs[0]

        progress_file = checkpoint_dir / "progress.json"
        if not await asyncio.to_thread(progress_file.exists):
            return None

        try:
            async with aiofiles.open(progress_file, "r", encoding="utf-8") as f:
                content = await f.read()
                progress = json.loads(content)

            return CheckpointInfo(
                checkpoint_dir=checkpoint_dir,
                file_hash=file_hash,
                completed_pages=progress.get("completed_pages", 0),
                total_pages=progress.get("total_pages", 0),
                status=progress.get("status", "processing"),
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read checkpoint progress: %s", exc)
            return None

    async def create_checkpoint(self, file_hash: str, filename: str, total_pages: int) -> Path:
        """Create a new checkpoint directory and metadata."""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        checkpoint_dir = self.base_dir / f"{file_hash}-{timestamp}"
        await asyncio.to_thread(checkpoint_dir.mkdir, parents=True, exist_ok=True)

        pages_dir = checkpoint_dir / "pages"
        await asyncio.to_thread(pages_dir.mkdir, exist_ok=True)

        # Create metadata file
        metadata = {
            "file_hash": file_hash,
            "original_filename": filename,
            "total_pages": total_pages,
            "created_at": timestamp,
        }
        async with aiofiles.open(checkpoint_dir / "metadata.json", "w", encoding="utf-8") as f:
            await f.write(json.dumps(metadata, indent=2))

        # Create initial progress file
        progress = {
            "completed_pages": 0,
            "total_pages": total_pages,
            "status": "processing",
        }
        async with aiofiles.open(checkpoint_dir / "progress.json", "w", encoding="utf-8") as f:
            await f.write(json.dumps(progress, indent=2))

        return checkpoint_dir

    async def save_page(self, checkpoint_dir: Path, page_number: int, text: str) -> None:
        """Save a single page's extracted text."""
        pages_dir = checkpoint_dir / "pages"
        page_file = pages_dir / f"page_{page_number:04d}.txt"
        async with aiofiles.open(page_file, "w", encoding="utf-8") as f:
            await f.write(text)

    async def load_page(self, checkpoint_dir: Path, page_number: int) -> str | None:
        """Load a single page's extracted text. Returns None if not found."""
        pages_dir = checkpoint_dir / "pages"
        page_file = pages_dir / f"page_{page_number:04d}.txt"
        if not await asyncio.to_thread(page_file.exists):
            return None
        try:
            async with aiofiles.open(page_file, "r", encoding="utf-8") as f:
                return await f.read()
        except OSError:
            return None

    async def update_progress(self, checkpoint_dir: Path, completed_pages: int, total_pages: int, status: str = "processing") -> None:
        """Update the progress file with current state."""
        progress = {
            "completed_pages": completed_pages,
            "total_pages": total_pages,
            "status": status,
            "last_updated": datetime.now().isoformat(),
        }
        async with aiofiles.open(checkpoint_dir / "progress.json", "w", encoding="utf-8") as f:
            await f.write(json.dumps(progress, indent=2))

    async def load_all_pages(self, checkpoint_dir: Path, total_pages: int) -> list[str]:
        """Load all pages from checkpoint. Returns list with empty strings for missing pages."""
        pages = []
        for page_num in range(1, total_pages + 1):
            text = await self.load_page(checkpoint_dir, page_num)
            pages.append(text if text is not None else "")
        return pages

    async def get_completed_page_numbers(self, checkpoint_dir: Path) -> set[int]:
        """Get set of page numbers that have been completed."""
        pages_dir = checkpoint_dir / "pages"
        if not await asyncio.to_thread(pages_dir.exists):
            return set()

        completed = set()
        page_files = await asyncio.to_thread(lambda: list(pages_dir.glob("page_*.txt")))
        for page_file in page_files:
            try:
                # Extract page number from filename like "page_0001.txt"
                page_num = int(page_file.stem.split("_")[1])
                completed.add(page_num)
            except (ValueError, IndexError):
                continue

        return completed

    async def save_failed_pages(self, checkpoint_dir: Path, failed_pages: dict[int, dict]) -> None:
        """
        Save failed pages metadata to checkpoint.
        
        Args:
            checkpoint_dir: Checkpoint directory path
            failed_pages: Dict mapping page_number -> {reason: str, attempts: int, last_tried: str}
        """
        failed_file = checkpoint_dir / "failed_pages.json"
        async with aiofiles.open(failed_file, "w", encoding="utf-8") as f:
            # Convert int keys to strings for JSON
            await f.write(json.dumps({str(k): v for k, v in failed_pages.items()}, indent=2))

    async def load_failed_pages(self, checkpoint_dir: Path) -> dict[int, dict]:
        """
        Load failed pages metadata from checkpoint.
        
        Returns:
            Dict mapping page_number -> {reason: str, attempts: int, last_tried: str}
        """
        failed_file = checkpoint_dir / "failed_pages.json"
        if not await asyncio.to_thread(failed_file.exists):
            return {}

        try:
            async with aiofiles.open(failed_file, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)
                # Convert string keys back to ints
                return {int(k): v for k, v in data.items()}
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("Failed to read failed_pages.json: %s", exc)
            return {}
