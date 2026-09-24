from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks):
        self.chunks = chunks

        # Tokenize each document chunk
        self.tokenized_chunks = [
            chunk["text"].lower().split()
            for chunk in chunks
        ]

        # Create BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_chunks
        )

    def search(self, query, top_k=3):

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for index in ranked_indices:

            results.append({
                "text": self.chunks[index]["text"],
                "source": self.chunks[index]["metadata"]["source"],
                "page": self.chunks[index]["metadata"]["page"],
                "chunk": self.chunks[index]["metadata"]["chunk"],
                "score": float(scores[index])
            })

        return results