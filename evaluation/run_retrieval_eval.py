import json
import sys
from pathlib import Path

# Allow Python to import from the app directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from retrieval.hybrid_retriever import HybridRetriever


QUESTIONS_FILE = (
    PROJECT_ROOT / "evaluation" / "questions.json"
)


def load_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    questions = load_questions()

    retriever = HybridRetriever()

    print("\n==============================")
    print("RAG RETRIEVAL EVALUATION")
    print("==============================")

    for question in questions:

        question_id = question["id"]
        query = question["question"]

        print("\n------------------------------")
        print(f"Question ID: {question_id}")
        print(f"Question: {query}")
        print("------------------------------")

        results = retriever.search(
            query,
            top_k=3
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(f"\nRank {rank}")

            print(
                "Rerank score:",
                round(
                    result["rerank_score"],
                    4
                )
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

            print(
                "Text:",
                result["text"][:250]
                    .replace("\n", " ")
            )


if __name__ == "__main__":
    main()