def precision_at_k(retrieved_pages, relevant_pages, k):
    retrieved = retrieved_pages[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(
        1 for page in retrieved
        if page in relevant_pages
    )

    return relevant_count / len(retrieved)


def recall_at_k(retrieved_pages, relevant_pages, k):
    retrieved = retrieved_pages[:k]

    if not relevant_pages:
        return 0.0

    relevant_count = sum(
        1 for page in retrieved
        if page in relevant_pages
    )

    return relevant_count / len(relevant_pages)


def reciprocal_rank(retrieved_pages, relevant_pages):
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant_pages:
            return 1 / rank

    return 0.0