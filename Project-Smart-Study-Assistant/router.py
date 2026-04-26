"""Route queries to the right handler."""
from retriever import get_llm

ROUTE_CATEGORIES = ["study_question", "summarize", "flashcards", "quiz", "general"]


def classify_query(question: str) -> str:
    """Classify a user query into a category."""
    llm = get_llm()

    prompt = (
        f"Classify this question into EXACTLY ONE of these categories: "
        f"study_question, summarize, flashcards, quiz, general.\n\n"
        f"Categories:\n"
        f"- study_question: needs information from study notes\n"
        f"- summarize: user wants a topic summary\n"
        f"- flashcards: user wants flashcards generated\n"
        f"- quiz: user wants to be quizzed\n"
        f"- general: general question that doesn't need notes\n\n"
        f"Question: {question}\n\n"
        f"Reply with ONLY the category name in lowercase, nothing else."
    )
    result = llm.invoke(prompt).content.strip().lower()
    # Validate — fall back to study_question if LLM returns unexpected text
    return result if result in ROUTE_CATEGORIES else "study_question"


def route_query(question: str, rag_chain, tools: dict) -> str:
    """Route a query to the appropriate handler."""
    category = classify_query(question)
    print(f"  Routed to: {category}")

    if category == "study_question":
        return rag_chain.invoke(question)
    elif category == "summarize":
        result = tools["summarize"].invoke(question)
        return result.content if hasattr(result, "content") else str(result)
    elif category == "flashcards":
        result = tools["flashcards"].invoke(question)
        return result.content if hasattr(result, "content") else str(result)
    elif category == "quiz":
        result = tools["quiz"].invoke(question)
        return result.content if hasattr(result, "content") else str(result)
    else:  # general
        return get_llm().invoke(question).content
