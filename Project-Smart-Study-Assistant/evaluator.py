"""Evaluate and improve responses using self-reflection."""
from retriever import get_llm


def critique_response(question: str, answer: str) -> str:
    """Ask the LLM to critique its own answer (Session 8: Self-Reflection)."""
    llm = get_llm()

    prompt = (
        f"Critique this answer for accuracy, completeness, and clarity.\n"
        f"Question: {question}\n\nAnswer: {answer}\n\n"
        f"Provide specific feedback on what's missing or unclear."
    )
    return llm.invoke(prompt).content


def refine_response(question: str, answer: str, critique: str) -> str:
    """Refine the answer based on critique (Session 8: Self-Refine pattern)."""
    llm = get_llm()

    prompt = (
        f"Improve this answer based on the critique.\n"
        f"Question: {question}\n\nOriginal answer: {answer}\n\nCritique: {critique}\n\n"
        f"Provide an improved answer that addresses the critique."
    )
    return llm.invoke(prompt).content


def self_refine(question: str, answer: str, max_rounds: int = 2) -> str:
    """Run the Generate -> Critique -> Refine loop."""
    current_answer = answer
    for i in range(max_rounds):
        print(f"  Refinement round {i+1}...")
        critique = critique_response(question, current_answer)
        if "no issues" in critique.lower() or "looks good" in critique.lower():
            print(f"  Answer approved after {i+1} round(s)")
            break
        current_answer = refine_response(question, current_answer, critique)
    return current_answer


def precision_at_k(retrieved: list, relevant: list, k: int) -> float:
    """Calculate Precision@K (Session 12: Eval metrics)."""
    retrieved_k = retrieved[:k]
    count = sum(1 for doc in retrieved_k if doc in relevant)
    return count / k if k > 0 else 0


def recall_at_k(retrieved: list, relevant: list, k: int) -> float:
    """Calculate Recall@K."""
    retrieved_k = retrieved[:k]
    count = sum(1 for doc in retrieved_k if doc in relevant)
    total = len(relevant)
    return count / total if total > 0 else 0


def f1_at_k(retrieved: list, relevant: list, k: int) -> float:
    """Calculate F1@K (harmonic mean of precision and recall)."""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0
