"""Route queries to the right handler."""
from retriever import get_llm

ROUTE_CATEGORIES = ["study_question", "summarize", "flashcards", "quiz", "general"]


def classify_query(question: str) -> str:
    """Classify a user query into a category."""
    llm = get_llm()

    # TODO 17: Ask the LLM to classify the question into one of ROUTE_CATEGORIES
    # study_question = needs RAG retrieval from notes
    # summarize = wants a topic summary
    # flashcards = wants flashcards generated
    # quiz = wants a quiz
    # general = general question, no retrieval needed
    # Return ONLY the category name
    pass


def route_query(question: str, rag_chain, tools: dict) -> str:
    """Route a query to the appropriate handler."""
    category = classify_query(question)
    print(f"  Routed to: {category}")

    # TODO 18: Based on category, call the right handler:
    # "study_question" -> use rag_chain.invoke(question)
    # "summarize" -> use tools["summarize"].invoke(question)
    # "flashcards" -> use tools["flashcards"].invoke(question)
    # "quiz" -> use tools["quiz"].invoke(question)
    # "general" -> use get_llm().invoke(question).content
    pass
