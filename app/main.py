from retrieval.hybrid_retriever import HybridRetriever


retriever = HybridRetriever()


query = "What projects are recommended?"


results = retriever.search(
    query,
    top_k=3
)


print("\n==============================")
print("DENSE RETRIEVAL")
print("==============================")


for index, result in enumerate(
    results["dense"],
    start=1
):

    print(f"\nRESULT {index}")

    print("Score:", result["score"])
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Chunk:", result["chunk"])

    print("Text:")
    print(result["text"][:300])


print("\n==============================")
print("BM25 RETRIEVAL")
print("==============================")


for index, result in enumerate(
    results["bm25"],
    start=1
):

    print(f"\nRESULT {index}")

    print("Score:", result["score"])
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Chunk:", result["chunk"])

    print("Text:")
    print(result["text"][:300])