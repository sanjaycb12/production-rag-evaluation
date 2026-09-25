from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, documents, top_k=3):

        if not documents:
            return []

        pairs = [
            [query, document["text"]]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked_documents = []

        for document, score in zip(
            documents,
            scores
        ):

            result = document.copy()

            result["rerank_score"] = float(score)

            ranked_documents.append(result)

        ranked_documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return ranked_documents[:top_k]