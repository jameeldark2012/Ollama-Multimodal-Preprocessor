"""Checkpoint manager for OCR processing with file-based resume capability."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

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

    def calculate_file_hash(self, data: bytes) -> str:
        """Calculate SHA256 hash of file content (first 16 hex chars)."""
        return hashlib.sha256(data).hexdigest()[:16]

    def find_checkpoint(self, file_hash: str) -> CheckpointInfo | None:
        """Find existing checkpoint by file hash. Returns most recent if multiple exist."""
        matching_dirs = list(self.base_dir.glob(f"{file_hash}-*"))
        if not matching_dirs:
            return None

        # Sort by timestamp (newest first) and take the most recent
        matching_dirs.sort(reverse=True)
        checkpoint_dir = matching_dirs[0]

        progress_file = checkpoint_dir / "progress.json"
        if not progress_file.exists():
            return None

        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                progress = json.load(f)

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

    def create_checkpoint(self, file_hash: str, filename: str, total_pages: int) -> Path:
        """Create a new checkpoint directory and metadata."""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        checkpoint_dir = self.base_dir / f"{file_hash}-{timestamp}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        pages_dir = checkpoint_dir / "pages"
        pages_dir.mkdir(exist_ok=True)

        # Create metadata file
        metadata = {
            "file_hash": file_hash,
            "original_filename": filename,
            "total_pages": total_pages,
            "created_at": timestamp,
        }
        with open(checkpoint_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Create initial progress file
        progress = {
            "completed_pages": 0,
            "total_pages": total_pages,
            "status": "processing",
        }
        with open(checkpoint_dir / "progress.json", "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

        return checkpoint_dir

    def save_page(self, checkpoint_dir: Path, page_number: int, text: str) -> None:
        """Save a single page's extracted text."""
        pages_dir = checkpoint_dir / "pages"
        page_file = pages_dir / f"page_{page_number:04d}.txt"
        with open(page_file, "w", encoding="utf-8") as f:
            f.write(text)

    def load_page(self, checkpoint_dir: Path, page_number: int) -> str | None:
        """Load a single page's extracted text. Returns None if not found."""
        pages_dir = checkpoint_dir / "pages"
        page_file = pages_dir / f"page_{page_number:04d}.txt"
        if not page_file.exists():
            return None
        try:
            with open(page_file, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    def update_progress(self, checkpoint_dir: Path, completed_pages: int, total_pages: int, status: str = "processing") -> None:
        """Update the progress file with current state."""
        progress = {
            "completed_pages": completed_pages,
            "total_pages": total_pages,
            "status": status,
            "last_updated": datetime.now().isoformat(),
        }
        with open(checkpoint_dir / "progress.json", "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

    def load_all_pages(self, checkpoint_dir: Path, total_pages: int) -> list[str]:
        """Load all pages from checkpoint. Returns list with empty strings for missing pages."""
        pages = []
        for page_num in range(1, total_pages + 1):
            text = self.load_page(checkpoint_dir, page_num)
            pages.append(text if text is not None else "")
        return pages

    def get_completed_page_numbers(self, checkpoint_dir: Path) -> set[int]:
        """Get set of page numbers that have been completed."""
        pages_dir = checkpoint_dir / "pages"
        if not pages_dir.exists():
            return set()

        completed = set()
        for page_file in pages_dir.glob("page_*.txt"):
            try:
                # Extract page number from filename like "page_0001.txt"
                page_num = int(page_file.stem.split("_")[1])
                completed.add(page_num)
            except (ValueError, IndexError):
                continue

        return completed
