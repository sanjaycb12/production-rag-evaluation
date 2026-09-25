import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "app")
)

from retrieval.hybrid_retriever import HybridRetriever
from metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank
)


QUESTIONS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "ground_truth.json"
)


def load_json(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def main():

    questions = load_json(
        QUESTIONS_FILE
    )

    ground_truth = load_json(
        GROUND_TRUTH_FILE
    )

    ground_truth_map = {
        item["id"]: item["relevant_pages"]
        for item in ground_truth
    }

    retriever = HybridRetriever()

    all_precision = []
    all_recall = []
    all_mrr = []

    print("\n==============================")
    print("RAG RETRIEVAL EVALUATION")
    print("==============================")

    for question in questions:

        question_id = question["id"]
        query = question["question"]

        relevant_pages = ground_truth_map[
            question_id
        ]

        results = retriever.search(
            query,
            top_k=3
        )

        retrieved_pages = [
            result["page"]
            for result in results
        ]

        precision = precision_at_k(
            retrieved_pages,
            relevant_pages,
            3
        )

        recall = recall_at_k(
            retrieved_pages,
            relevant_pages,
            3
        )

        mrr = reciprocal_rank(
            retrieved_pages,
            relevant_pages
        )

        all_precision.append(precision)
        all_recall.append(recall)
        all_mrr.append(mrr)

        print("\n------------------------------")
        print(f"Question ID: {question_id}")
        print(f"Question: {query}")

        print(
            f"Relevant pages: {relevant_pages}"
        )

        print(
            f"Retrieved pages: {retrieved_pages}"
        )

        print(
            f"Precision@3: {precision:.3f}"
        )

        print(
            f"Recall@3: {recall:.3f}"
        )

        print(
            f"MRR: {mrr:.3f}"
        )

    avg_precision = (
        sum(all_precision)
        / len(all_precision)
    )

    avg_recall = (
        sum(all_recall)
        / len(all_recall)
    )

    avg_mrr = (
        sum(all_mrr)
        / len(all_mrr)
    )

    print("\n==============================")
    print("OVERALL RESULTS")
    print("==============================")

    print(
        f"Average Precision@3: "
        f"{avg_precision:.3f}"
    )

    print(
        f"Average Recall@3: "
        f"{avg_recall:.3f}"
    )

    print(
        f"Mean Reciprocal Rank: "
        f"{avg_mrr:.3f}"
    )


if __name__ == "__main__":
    main()