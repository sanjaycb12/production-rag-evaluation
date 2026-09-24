from ingestion.chunk_documents import create_chunks
from retrieval.embedding import EmbeddingModel
from retrieval.vector_store import VectorStore


# --------------------------------
# 1. Load and chunk documents
# --------------------------------

chunks = create_chunks("data/raw")

print(
    f"\nTotal chunks: {len(chunks)}"
)


# --------------------------------
# 2. Create embeddings
# --------------------------------

embedding_model = EmbeddingModel()

texts = [
    chunk["text"]
    for chunk in chunks
]

vectors = embedding_model.encode(texts)

print(
    f"Number of vectors: {len(vectors)}"
)

print(
    f"Vector dimension: {len(vectors[0])}"
)


# --------------------------------
# 3. Connect to Qdrant
# --------------------------------

vector_store = VectorStore()


# --------------------------------
# 4. Create collection
# --------------------------------

vector_store.create_collection(
    vector_size=len(vectors[0])
)


# --------------------------------
# 5. Insert vectors
# --------------------------------

vector_store.insert_documents(
    chunks,
    vectors
)