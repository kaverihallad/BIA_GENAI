"""Custom tools for the study assistant agent."""
from langchain_core.tools import tool
from retriever import get_llm


# TODO 10: Create a @tool called summarize_topic that takes a topic (str)
# and asks the LLM to create a concise summary
# Hint: Use get_llm().invoke() with a prompt like "Summarize this topic: {topic}"
@tool
def summarize_topic(topic: str) -> str:
    """Create a concise summary of a study topic."""
    llm = get_llm()
    prompt = f"Provide a concise 3-4 bullet point summary of this topic: {topic}. Focus on key concepts."
    return llm.invoke(prompt).content


# TODO 11: Create a @tool called generate_flashcards that takes content (str)
# and asks the LLM to create 5 Q&A flashcards from it
@tool
def generate_flashcards(content: str) -> str:
    """Generate study flashcards (Q&A pairs) from content."""
    llm = get_llm()
    prompt = f"Create 5 study flashcards (Q&A pairs) from this content: {content}. Format each as 'Q: [question]\nA: [answer]'"
    return llm.invoke(prompt).content


# TODO 12: Create a @tool called quiz_me that takes a topic (str)
# and generates a 3-question multiple choice quiz
@tool
def quiz_me(topic: str) -> str:
    """Generate a multiple-choice quiz on a topic."""
    llm = get_llm()
    prompt = f"Create a 3-question multiple choice quiz on this topic: {topic}. For each question, provide 4 options (A, B, C, D) and indicate the correct answer."
    return llm.invoke(prompt).content


def get_all_tools():
    """Return all available tools."""
    return [summarize_topic, generate_flashcards, quiz_me]
