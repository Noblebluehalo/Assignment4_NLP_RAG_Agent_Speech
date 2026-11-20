import os
from rag import get_or_build_vectorstore, get_llm, ask_syllabus, PERSIST_DIR, CONFIG_PATH

# Syllabus-specific evaluation questions covering specific requirements, policies, and course details
EVAL_QUESTIONS = [
    "What is the grading breakdown or weight distribution for assignments, exams, and projects in CPSC 254?",
    "What is the course policy on late submissions, attendance, or academic integrity?",
    "What are the prerequisites, required textbook, or instructor contact/office hours details for CPSC 254?",
    "What is the exact menu served at the campus cafeteria on Fridays?",  # Abstention test
]


def run_evaluation():
    print("=" * 75)
    print("PART 3.2: CPSC 254 SYLLABUS RAG EVALUATION")
    print("=" * 75)

    # Ensure index exists (builds once if rag.py wasn't run yet, then reuses saved index)
    if not (os.path.exists(PERSIST_DIR) and os.path.exists(CONFIG_PATH)):
        print("Saved index not found; building initial index first...")
        get_or_build_vectorstore()

    # Load saved index from disk
    vectorstore = get_or_build_vectorstore(force_rebuild=False)
    llm = get_llm()

    for idx, question in enumerate(EVAL_QUESTIONS, start=1):
        print("\n" + "-" * 75)
        print(f"QUESTION {idx}: {question}")
        print("-" * 75)

        result = ask_syllabus(question, vectorstore=vectorstore, llm=llm)

        print("Retrieved Supporting Chunks:")
        for c_idx, doc in enumerate(result["retrieved_docs"], start=1):
            page = doc.metadata.get("page", "?")
            display_page = page + 1 if isinstance(page, int) else page
            snippet = " ".join(doc.page_content.split())[:220]
            print(f"  [{c_idx}] Page {display_page}: \"{snippet}...\"")

        print(f"\nGenerated Answer:\n{result['answer']}\n")


if __name__ == "__main__":
    run_evaluation()
