from retrieval.hybrid_retriever import HybridRetriever


retriever = HybridRetriever()

query = "What projects are recommended?"

results = retriever.search(
    query,
    top_k=3
)


print("\n==============================")
print("RERANKED RESULTS")
print("==============================")


for index, result in enumerate(
    results,
    start=1
):

    print(f"\nRESULT {index}")

    print(
        "Rerank score:",
        result["rerank_score"]
    )

    print(
        "RRF score:",
        result.get("rrf_score")
    )

    print(
        "Source:",
        result["source"]
    )

    print(
        "Page:",
        result["page"]
    )

    print(
        "Chunk:",
        result["chunk"]
    )

    print("\nText:")

    print(
        result["text"][:500]
    )