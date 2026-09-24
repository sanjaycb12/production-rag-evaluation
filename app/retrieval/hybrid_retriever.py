from retrieval.retriever import Retriever
from retrieval.bm25 import BM25Retriever
from ingestion.chunk_documents import create_chunks


class HybridRetriever:

    def __init__(self):

        # Load document chunks
        self.chunks = create_chunks(
            "data/raw"
        )

        # Dense retriever
        self.dense_retriever = Retriever()

        # BM25 retriever
        self.bm25_retriever = BM25Retriever(
            self.chunks
        )

    def search(self, query, top_k=3):

        dense_results = self.dense_retriever.search(
            query,
            top_k=top_k
        )

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=top_k
        )

        return {
            "dense": dense_results,
            "bm25": bm25_results
        }