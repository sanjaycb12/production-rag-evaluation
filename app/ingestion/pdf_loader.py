from pathlib import Path
import fitz


def load_pdf(pdf_path: str):
    """
    Extract text from a PDF page by page.
    """

    pdf = fitz.open(pdf_path)

    documents = []

    for page_number, page in enumerate(pdf):
        text = page.get_text()

        documents.append({
            "text": text,
            "metadata": {
                "source": Path(pdf_path).name,
                "page": page_number + 1
            }
        })

    pdf.close()

    return documents