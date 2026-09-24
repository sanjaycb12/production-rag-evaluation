def chunk_document(document, chunk_size=500, overlap=50):
    text = document["text"]

    chunks = []

    start = 0
    chunk_number = 0

    while start < len(text):
        end = start + chunk_size

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **document["metadata"],
                    "chunk": chunk_number
                }
            })

        chunk_number += 1
        start += chunk_size - overlap

    return chunks