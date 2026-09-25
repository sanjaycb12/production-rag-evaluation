def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=60
):

    scores = {}
    documents = {}

    # Process dense results
    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        key = (
            result["source"],
            result["page"],
            result["chunk"]
        )

        scores[key] = scores.get(key, 0) + (
            1 / (k + rank)
        )

        documents[key] = result

    # Process BM25 results
    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        key = (
            result["source"],
            result["page"],
            result["chunk"]
        )

        scores[key] = scores.get(key, 0) + (
            1 / (k + rank)
        )

        documents[key] = result

    # Sort by combined RRF score
    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    results = []

    for key in ranked_keys:

        result = documents[key].copy()

        result["rrf_score"] = scores[key]

        results.append(result)

    return results