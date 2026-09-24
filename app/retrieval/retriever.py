from retrieval.embedding import EmbeddingModel
from retrieval.vector_store import VectorStore


class Retriever:

    def __init__(self):

        self.embedding_model = EmbeddingModel()

        self.vector_store = VectorStore()

    def search(self, query, top_k=3):

        # Convert the user's question into a vector
        query_vector = self.embedding_model.encode(
            [query]
        )[0]

        # Search Qdrant
        results = self.vector_store.client.query_points(
            collection_name=self.vector_store.collection_name,
            query=query_vector.tolist(),
            limit=top_k
        ).points

        retrieved_documents = []

        for result in results:

            retrieved_documents.append({
                "text": result.payload["text"],
                "source": result.payload["source"],
                "page": result.payload["page"],
                "chunk": result.payload["chunk"],
                "score": result.score
            })

        return retrieved_documents