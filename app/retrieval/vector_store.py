from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)


class VectorStore:

    def __init__(self):
        self.client = QdrantClient(
            url="http://localhost:6333"
        )

        self.collection_name = "production_rag"

    def create_collection(self, vector_size=384):

        collections = self.client.get_collections().collections

        collection_names = [
            collection.name
            for collection in collections
        ]

        if self.collection_name not in collection_names:

            self.client.create_collection(
                collection_name=self.collection_name,

                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

            print(
                f"Collection '{self.collection_name}' created."
            )

        else:

            print(
                f"Collection '{self.collection_name}' already exists."
            )

    def insert_documents(self, chunks, vectors):

        points = []

        for index, (chunk, vector) in enumerate(
            zip(chunks, vectors)
        ):

            point = PointStruct(
                id=index,
                vector=vector.tolist(),
                payload={
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "page": chunk["metadata"]["page"],
                    "chunk": chunk["metadata"]["chunk"]
                }
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(
            f"Inserted {len(points)} vectors into Qdrant."
        )