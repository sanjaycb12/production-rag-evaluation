from retrieval.retriever import Retriever
from retrieval.bm25 import BM25Retriever
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.reranker import Reranker
from ingestion.chunk_documents import create_chunks


class HybridRetriever:

    def __init__(self):

        self.chunks = create_chunks(
            "data/raw"
        )

        self.dense_retriever = Retriever()

        self.bm25_retriever = BM25Retriever(
            self.chunks
        )

        self.reranker = Reranker()

    def search(self, query, top_k=3):

        # Retrieve more candidates first
        retrieval_k = 10

        dense_results = (
            self.dense_retriever.search(
                query,
                top_k=retrieval_k
            )
        )

        bm25_results = (
            self.bm25_retriever.search(
                query,
                top_k=retrieval_k
            )
        )

        # Combine dense + BM25 rankings
        fused_results = reciprocal_rank_fusion(
            dense_results,
            bm25_results
        )

        # Rerank the combined candidates
        reranked_results = self.reranker.rerank(
            query,
            fused_results,
            top_k=top_k
        )

        return reranked_results