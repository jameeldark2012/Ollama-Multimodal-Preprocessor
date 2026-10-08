"""Quick test of Gemini backend integration."""
import asyncio
import sys
from pathlib import Path
from io import BytesIO

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.ocr import extract_attachment
from PIL import Image, ImageDraw, ImageFont


async def create_test_image() -> bytes:
    """Create a simple test image with text."""
    img = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some text
    text = "Hello from Gemini OCR Test!\nThis is a test image with multiple lines.\nLine 3 with numbers: 12345"
    try:
        # Try to use a nice font
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        # Fall back to default
        font = ImageFont.load_default()
    
    draw.text((20, 20), text, fill='black', font=font)
    
    # Convert to JPEG bytes
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=95)
    return buf.getvalue()


async def main():
    print(f"Testing Gemini backend...")
    print(f"Backend: {settings.backend}")
    print(f"Model: {settings.active_model}")
    print(f"API Key configured: {'Yes' if settings.gemini_api_key else 'No'}")
    print(f"Batch size: {settings.gemini_batch_size}")
    print(f"RPM limit: {settings.gemini_rpm_limit}")
    print(f"Fallback to local: {settings.gemini_fallback_to_local}")
    print()
    
    # Test 1: Text file
    print("=" * 60)
    print("TEST 1: Text file processing")
    print("=" * 60)
    test_text = b"Hello from Gemini test!"
    try:
        result, pages = await extract_attachment("test.txt", test_text)
        print(f"[OK] Text file test passed")
        print(f"Result: {result}")
        print(f"Pages: {pages}")
        print()
    except Exception as exc:
        print(f"[FAIL] Text file test failed: {exc}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Single image OCR
    print("=" * 60)
    print("TEST 2: Single image OCR with Gemini")
    print("=" * 60)
    try:
        test_image = await create_test_image()
        print(f"Created test image: {len(test_image)} bytes")
        
        print("Starting OCR...")
        text, pages = await extract_attachment("test_image.jpg", test_image)
        print(f"[OK] Image OCR completed successfully!")
        print(f"Pages processed: {pages}")
        print(f"Text length: {len(text)} characters")
        print(f"Extracted text:")
        print("-" * 60)
        print(text)
        print("-" * 60)
        print()
    except Exception as exc:
        print(f"[FAIL] Image OCR failed: {exc}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Check for PDF files
    print("=" * 60)
    print("TEST 3: PDF processing (if available)")
    print("=" * 60)
    sample_pdfs = list(Path(".").rglob("*.pdf"))
    if not sample_pdfs:
        print("No PDF files found. Skipping PDF test.")
        print("To test PDF processing, add a sample PDF to the project directory.")
    else:
        sample_pdf = sample_pdfs[0]
        print(f"Testing with: {sample_pdf}")
        print(f"File size: {sample_pdf.stat().st_size / 1024:.1f} KB")
        
        try:
            data = sample_pdf.read_bytes()
            print("Starting OCR...")
            text, pages = await extract_attachment(sample_pdf.name, data)
            print(f"[OK] PDF OCR completed successfully!")
            print(f"Pages processed: {pages}")
            print(f"Text length: {len(text)} characters")
            print(f"First 300 chars:")
            print("-" * 60)
            print(text[:300])
            print("-" * 60)
        except Exception as exc:
            print(f"[FAIL] PDF OCR failed: {exc}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
