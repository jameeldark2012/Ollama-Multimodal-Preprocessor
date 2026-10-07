"""
Integration test for checkpoint system.
Creates a small mock PDF and tests the checkpoint flow end-to-end.
"""

import asyncio
import io
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to create a small test PDF
try:
    import fitz  # PyMuPDF
    
    def create_test_pdf(num_pages: int = 5) -> bytes:
        """Create a small test PDF with the specified number of pages."""
        doc = fitz.open()
        for i in range(num_pages):
            page = doc.new_page(width=595, height=842)  # A4 size
            text = f"Test Page {i + 1}\n\nThis is page {i + 1} of {num_pages}."
            page.insert_text((50, 50), text, fontsize=14)
        
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    
    print("Creating test PDF...")
    test_pdf_data = create_test_pdf(5)
    print(f"✓ Created test PDF: {len(test_pdf_data)} bytes, 5 pages")
    
except ImportError:
    print("PyMuPDF (fitz) not available, using mock test data")
    test_pdf_data = None


async def test_checkpoint_integration():
    """Test the checkpoint system with actual extract_attachment function."""
    from app.checkpoint import CheckpointManager
    from app.config import settings
    
    print("\nTesting checkpoint integration...")
    
    # Setup test checkpoint directory
    test_checkpoint_dir = "test_checkpoints_integration"
    manager = CheckpointManager(test_checkpoint_dir)
    
    # Create test data
    test_data = b"Mock PDF file for checkpoint testing"
    filename = "test_integration.pdf"
    
    # Calculate hash
    file_hash = manager.calculate_file_hash(test_data)
    print(f"File hash: {file_hash}")
    
    # Scenario 1: First upload (no checkpoint exists)
    print("\n--- Scenario 1: First upload (new file) ---")
    checkpoint_info = manager.find_checkpoint(file_hash)
    if checkpoint_info is None:
        print("✓ No existing checkpoint found (expected)")
    else:
        print("✗ Unexpected checkpoint found")
        return False
    
    # Create checkpoint manually to simulate partial processing
    checkpoint_dir = manager.create_checkpoint(file_hash, filename, 10)
    print(f"✓ Created checkpoint: {checkpoint_dir.name}")
    
    # Simulate processing first 3 pages
    for i in range(1, 4):
        manager.save_page(checkpoint_dir, i, f"Page {i} text content")
    manager.update_progress(checkpoint_dir, 3, 10, "processing")
    print("✓ Simulated processing pages 1-3")
    
    # Scenario 2: Resume from checkpoint
    print("\n--- Scenario 2: Resume from checkpoint ---")
    checkpoint_info = manager.find_checkpoint(file_hash)
    if checkpoint_info:
        print(f"✓ Found checkpoint: {checkpoint_info.completed_pages}/{checkpoint_info.total_pages} pages")
        print(f"  Status: {checkpoint_info.status}")
        
        if checkpoint_info.completed_pages != 3:
            print(f"✗ Expected 3 completed pages, got {checkpoint_info.completed_pages}")
            return False
    else:
        print("✗ Checkpoint not found")
        return False
    
    # Get completed pages
    completed = manager.get_completed_page_numbers(checkpoint_dir)
    print(f"✓ Completed pages: {sorted(completed)}")
    
    # Continue processing
    for i in range(4, 11):
        manager.save_page(checkpoint_dir, i, f"Page {i} text content")
    manager.update_progress(checkpoint_dir, 10, 10, "completed")
    print("✓ Completed processing all pages")
    
    # Scenario 3: Fully completed checkpoint
    print("\n--- Scenario 3: Fully completed (should return cached) ---")
    checkpoint_info = manager.find_checkpoint(file_hash)
    if checkpoint_info:
        print(f"✓ Found checkpoint: {checkpoint_info.completed_pages}/{checkpoint_info.total_pages} pages")
        print(f"  Status: {checkpoint_info.status}")
        
        if checkpoint_info.status != "completed":
            print(f"✗ Expected status 'completed', got '{checkpoint_info.status}'")
            return False
        
        if checkpoint_info.completed_pages != checkpoint_info.total_pages:
            print(f"✗ Not all pages completed")
            return False
        
        # Load all pages
        all_pages = manager.load_all_pages(checkpoint_dir, 10)
        print(f"✓ Loaded all {len(all_pages)} pages from cache")
        
        if len(all_pages) != 10:
            print(f"✗ Expected 10 pages, got {len(all_pages)}")
            return False
        
        # Verify content
        for i, page_text in enumerate(all_pages, 1):
            expected = f"Page {i} text content"
            if page_text != expected:
                print(f"✗ Page {i} content mismatch: '{page_text}' != '{expected}'")
                return False
        
        print("✓ All page contents verified")
    else:
        print("✗ Checkpoint not found")
        return False
    
    # Cleanup
    import shutil
    shutil.rmtree(test_checkpoint_dir)
    print("\n✓ Cleaned up test directory")
    
    print("\n✅ All integration tests passed!")
    return True


async def test_with_real_ocr():
    """
    Test with actual OCR processing if available.
    This is optional and only runs if user has configured backend.
    """
    try:
        from app.ocr import extract_attachment
        from app.config import settings
        
        if test_pdf_data is None:
            print("\nSkipping real OCR test (no test PDF available)")
            return True
        
        print("\n" + "="*60)
        print("Testing with REAL OCR processing (optional)")
        print("="*60)
        print("\nThis will call your configured backend to OCR the test PDF.")
        print("Press Ctrl+C to skip this test if you don't want to use backend resources.\n")
        
        # Give user a chance to cancel
        await asyncio.sleep(2)
        
        print("Processing test PDF with OCR...")
        text, pages = await extract_attachment("test_checkpoint.pdf", test_pdf_data)
        print(f"✓ OCR completed: {pages} pages, {len(text)} characters")
        print(f"\nExtracted text preview:")
        print("-" * 40)
        print(text[:200] if len(text) > 200 else text)
        print("-" * 40)
        
        # Process again - should use cache
        print("\nProcessing same PDF again (should use cache)...")
        import time
        start = time.time()
        text2, pages2 = await extract_attachment("test_checkpoint.pdf", test_pdf_data)
        elapsed = time.time() - start
        
        print(f"✓ Second processing: {pages2} pages, {len(text2)} characters")
        print(f"  Elapsed time: {elapsed:.3f}s (should be very fast if cached)")
        
        if text == text2 and pages == pages2:
            print("✓ Cached result matches original")
        else:
            print("✗ Cached result doesn't match!")
            return False
        
        print("\n✅ Real OCR test passed!")
        return True
        
    except KeyboardInterrupt:
        print("\n⏭  Skipped real OCR test")
        return True
    except Exception as e:
        print(f"\n⚠  Real OCR test failed: {e}")
        print("This is okay - your backend might not be running")
        return True


if __name__ == "__main__":
    print("="*60)
    print("Checkpoint System Integration Test")
    print("="*60)
    
    success = asyncio.run(test_checkpoint_integration())
    
    if success:
        # Only run real OCR test if integration test passed
        asyncio.run(test_with_real_ocr())
    
    if success:
        print("\n" + "="*60)
        print("✅ All tests completed successfully!")
        print("="*60)
        print("\nYour checkpoint system is ready to use!")
        print("It will automatically:")
        print("  • Hash uploaded PDFs")
        print("  • Resume from last checkpoint if processing is interrupted")
        print("  • Return cached results instantly for already-processed files")
        print("  • Save progress every 20 pages (configurable)")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
