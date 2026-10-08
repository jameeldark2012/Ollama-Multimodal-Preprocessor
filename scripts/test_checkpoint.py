"""Test script to verify checkpoint functionality without running the live server."""

import asyncio
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.checkpoint import CheckpointManager


async def test_checkpoint_manager():
    """Test the checkpoint manager basic operations."""
    print("Testing CheckpointManager...")

    # Create a test checkpoint manager
    test_dir = Path("test_checkpoints")
    manager = CheckpointManager(str(test_dir))

    # Test 1: Calculate file hash
    test_data = b"This is a test PDF content"
    file_hash = await manager.calculate_file_hash(test_data)
    print(f"✓ File hash calculated: {file_hash}")
    assert len(file_hash) == 16, "Hash should be 16 characters"

    # Test 2: Create checkpoint
    checkpoint_dir = await manager.create_checkpoint(file_hash, "test.pdf", 10)
    print(f"✓ Checkpoint created: {checkpoint_dir}")
    assert checkpoint_dir.exists(), "Checkpoint directory should exist"
    assert (checkpoint_dir / "metadata.json").exists(), "Metadata file should exist"
    assert (checkpoint_dir / "progress.json").exists(), "Progress file should exist"
    assert (checkpoint_dir / "pages").exists(), "Pages directory should exist"

    # Test 3: Save pages
    for i in range(1, 6):
        await manager.save_page(checkpoint_dir, i, f"Page {i} content")
    print(f"✓ Saved 5 pages")

    # Test 4: Load page
    page_text = await manager.load_page(checkpoint_dir, 3)
    assert page_text == "Page 3 content", "Loaded page content should match"
    print(f"✓ Loaded page 3: '{page_text}'")

    # Test 5: Get completed page numbers
    completed = await manager.get_completed_page_numbers(checkpoint_dir)
    assert completed == {1, 2, 3, 4, 5}, "Completed pages should be 1-5"
    print(f"✓ Completed pages: {sorted(completed)}")

    # Test 6: Update progress
    await manager.update_progress(checkpoint_dir, 5, 10, "processing")
    print(f"✓ Progress updated")

    # Test 7: Find checkpoint
    found = await manager.find_checkpoint(file_hash)
    assert found is not None, "Should find the checkpoint"
    assert found.file_hash == file_hash, "Hash should match"
    assert found.completed_pages == 5, "Should have 5 completed pages"
    assert found.total_pages == 10, "Should have 10 total pages"
    assert found.status == "processing", "Status should be processing"
    print(f"✓ Checkpoint found: {found.completed_pages}/{found.total_pages} pages, status={found.status}")

    # Test 8: Load all pages
    all_pages = await manager.load_all_pages(checkpoint_dir, 10)
    assert len(all_pages) == 10, "Should have 10 pages"
    assert all_pages[0] == "Page 1 content", "First page should match"
    assert all_pages[4] == "Page 5 content", "Fifth page should match"
    assert all_pages[5] == "", "Sixth page should be empty"
    print(f"✓ Loaded all pages: {len(all_pages)} pages")

    # Test 9: Mark as completed
    for i in range(6, 11):
        await manager.save_page(checkpoint_dir, i, f"Page {i} content")
    await manager.update_progress(checkpoint_dir, 10, 10, "completed")
    found = await manager.find_checkpoint(file_hash)
    assert found.status == "completed", "Status should be completed"
    assert found.completed_pages == 10, "All pages should be completed"
    print(f"✓ Checkpoint marked as completed")

    # Test 10: Different file hash should not find checkpoint
    different_data = b"Different content"
    different_hash = await manager.calculate_file_hash(different_data)
    found = await manager.find_checkpoint(different_hash)
    assert found is None, "Should not find checkpoint for different hash"
    print(f"✓ Different hash correctly returns None")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print(f"✓ Cleaned up test directory")

    print("\n✅ All checkpoint manager tests passed!")


async def test_checkpoint_with_mock_pdf():
    """Test checkpoint system with a simulated PDF processing flow."""
    print("\nTesting checkpoint system with simulated PDF processing...")

    test_dir = Path("test_checkpoints_sim")
    manager = CheckpointManager(str(test_dir))

    # Simulate a PDF upload
    pdf_data = b"Mock PDF content for testing checkpoint resume"
    file_hash = await manager.calculate_file_hash(pdf_data)
    filename = "test_document.pdf"
    total_pages = 50

    print(f"Simulating PDF: {filename} ({total_pages} pages)")
    print(f"File hash: {file_hash}")

    # First upload: process 25 pages then "crash"
    print("\n--- First upload (processing 25 pages then stopping) ---")
    checkpoint_info = await manager.find_checkpoint(file_hash)
    if checkpoint_info is None:
        checkpoint_dir = await manager.create_checkpoint(file_hash, filename, total_pages)
        print(f"Created new checkpoint: {checkpoint_dir.name}")
    else:
        checkpoint_dir = checkpoint_info.checkpoint_dir
        print(f"Found existing checkpoint: {checkpoint_dir.name}")

    for page_num in range(1, 26):
        # Simulate OCR processing
        page_text = f"Simulated OCR text for page {page_num}"
        await manager.save_page(checkpoint_dir, page_num, page_text)
        
        if page_num % 20 == 0:
            await manager.update_progress(checkpoint_dir, page_num, total_pages, "processing")
            print(f"Checkpoint saved at page {page_num}")

    await manager.update_progress(checkpoint_dir, 25, total_pages, "processing")
    print(f"Stopped at page 25/50")

    # Second upload: resume from page 26
    print("\n--- Second upload (resuming from page 26) ---")
    checkpoint_info = await manager.find_checkpoint(file_hash)
    assert checkpoint_info is not None, "Should find existing checkpoint"
    print(f"Found checkpoint: {checkpoint_info.completed_pages}/{checkpoint_info.total_pages} pages completed")

    completed_pages = await manager.get_completed_page_numbers(checkpoint_info.checkpoint_dir)
    print(f"Already completed pages: {len(completed_pages)} pages")

    # Resume processing
    for page_num in range(26, 51):
        if page_num in completed_pages:
            print(f"Skipping page {page_num} (already done)")
            continue

        # Simulate OCR processing
        page_text = f"Simulated OCR text for page {page_num}"
        await manager.save_page(checkpoint_dir, page_num, page_text)

        if page_num % 20 == 0:
            completed = len(completed_pages) + (page_num - 25)
            await manager.update_progress(checkpoint_dir, completed, total_pages, "processing")
            print(f"Checkpoint saved at page {page_num}")

    await manager.update_progress(checkpoint_dir, total_pages, total_pages, "completed")
    print(f"Processing completed: {total_pages}/{total_pages} pages")

    # Third upload: should return immediately (already completed)
    print("\n--- Third upload (already completed, should be instant) ---")
    checkpoint_info = await manager.find_checkpoint(file_hash)
    assert checkpoint_info.status == "completed", "Status should be completed"
    assert checkpoint_info.completed_pages == total_pages, "All pages should be done"
    print(f"Checkpoint status: {checkpoint_info.status}")
    print(f"✓ Would return cached result immediately!")

    # Load all pages
    all_pages = await manager.load_all_pages(checkpoint_dir, total_pages)
    assert len(all_pages) == total_pages, "Should have all pages"
    assert all(page for page in all_pages), "All pages should have content"
    print(f"✓ All {len(all_pages)} pages loaded successfully")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print(f"✓ Cleaned up test directory")

    print("\n✅ Simulated PDF checkpoint test passed!")


if __name__ == "__main__":
    asyncio.run(test_checkpoint_manager())
    asyncio.run(test_checkpoint_with_mock_pdf())
    print("\n" + "=" * 60)
    print("All tests passed! Checkpoint system is working correctly.")
    print("=" * 60)
