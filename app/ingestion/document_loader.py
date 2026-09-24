from pathlib import Path

from ingestion.pdf_loader import load_pdf
from ingestion.text_cleaner import clean_text


def load_all_pdfs(data_directory: str):
    documents = []

    pdf_files = Path(data_directory).glob("*.pdf")

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        pdf_documents = load_pdf(str(pdf_file))

        for document in pdf_documents:
            document["text"] = clean_text(document["text"])

        documents.extend(pdf_documents)

    return documents