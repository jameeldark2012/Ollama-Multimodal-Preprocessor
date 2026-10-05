import argparse
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app

DEFAULT_PDF = Path(r"E:\Downloads\Noor-Book.com  حلية الأولياء وطبقات الأصفياء-3.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OCR backend against the sample PDF.")
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    pdf_path = parser.parse_args().pdf
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    with pdf_path.open("rb") as source, TestClient(app) as client:
        response = client.post(
            "/v1/ocr",
            files={"file": (pdf_path.name, source, "application/pdf")},
        )
    response.raise_for_status()
    result = response.json()
    output_path = pdf_path.with_name(f"{pdf_path.stem} - OCR.txt")
    output_path.write_text(result["text"], encoding="utf-8")
    print(f"Pages processed: {result['pages_processed']}")
    print(f"Characters extracted: {len(result['text'])}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()