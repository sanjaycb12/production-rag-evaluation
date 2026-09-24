from ingestion.document_loader import load_all_pdfs
from ingestion.chunker import chunk_document


def create_chunks(data_directory: str):
    documents = load_all_pdfs(data_directory)

    all_chunks = []

    for document in documents:
        chunks = chunk_document(document)

        all_chunks.extend(chunks)

    return all_chunks