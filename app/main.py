from retrieval.retriever import Retriever


retriever = Retriever()


query = "What projects are recommended?"


results = retriever.search(
    query,
    top_k=3
)


print("\n==============================")
print("SEARCH RESULTS")
print("==============================")


for index, result in enumerate(results, start=1):

    print(f"\nRESULT {index}")

    print("Score:", result["score"])

    print("Source:", result["source"])

    print("Page:", result["page"])

    print("Chunk:", result["chunk"])

    print("Text:")

    print(result["text"][:500])