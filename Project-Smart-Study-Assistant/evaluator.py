"""Evaluate and improve responses using self-reflection."""
from retriever import get_llm


def critique_response(question: str, answer: str) -> str:
    """Ask the LLM to critique its own answer (Session 8: Self-Reflection)."""
    llm = get_llm()

    # TODO 13: Create a prompt that asks the LLM to critique the answer
    # Check for: accuracy, completeness, clarity
    # Return the critique as a string
    # Hint: llm.invoke("Critique this answer for accuracy and completeness: ...")
    pass


def refine_response(question: str, answer: str, critique: str) -> str:
    """Refine the answer based on critique (Session 8: Self-Refine pattern)."""
    llm = get_llm()

    # TODO 14: Create a prompt that asks the LLM to improve the answer
    # based on the critique. Return the improved answer.
    pass


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
    # TODO 15: Implement precision@k
    # retrieved_k = retrieved[:k]
    # Count how many of retrieved_k are in relevant
    # Return count / k
    pass


def recall_at_k(retrieved: list, relevant: list, k: int) -> float:
    """Calculate Recall@K."""
    # TODO 16: Implement recall@k
    pass


def f1_at_k(retrieved: list, relevant: list, k: int) -> float:
    """Calculate F1@K (harmonic mean of precision and recall)."""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0
