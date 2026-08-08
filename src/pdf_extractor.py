import pymupdf
from pathlib import Path


class PDFExtractor:
    """
    Extract text and page metadata from PDF documents.
    """

    def extract(self, pdf_path):
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = pymupdf.open(pdf_path)

        pages = []

        for page_number, page in enumerate(doc, start=1):
            pages.append({
                "page": page_number,
                "text": page.get_text().strip()
            })

        doc.close()

        return {
            "filename": pdf_path.name,
            "total_pages": len(pages),
            "pages": pages
        }

    def extract_text(self, pdf_path):
        document = self.extract(pdf_path)
        return "\n".join(page["text"] for page in document["pages"])