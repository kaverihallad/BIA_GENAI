"""The main study assistant agent using LangGraph."""
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from config import LLM_MODEL, LLM_TEMPERATURE
from tools import get_all_tools

AGENT_PROMPT = """You are a Smart Study Assistant. You help students learn effectively.

You have access to these tools:
- summarize_topic: Create concise summaries of study topics
- generate_flashcards: Generate Q&A flashcards from content
- quiz_me: Generate multiple-choice quizzes on topics

When a student asks a question:
1. If they want a summary, use summarize_topic
2. If they want flashcards, use generate_flashcards
3. If they want a quiz, use quiz_me
4. For general questions, answer directly from your knowledge

Be encouraging and helpful. Keep explanations clear for beginners."""


def create_study_agent():
    """Create the study assistant agent."""
    # TODO 19: Initialize the LLM with LLM_MODEL and LLM_TEMPERATURE
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    # TODO 20: Get all tools from get_all_tools()
    tools = get_all_tools()

    # TODO 21: Create the agent using create_react_agent
    agent = create_react_agent(model=llm, tools=tools, prompt=AGENT_PROMPT)

    return agent


def chat_with_agent(agent, message: str) -> str:
    """Send a message to the agent and get a response."""
    response = agent.invoke({"messages": [{"role": "user", "content": message}]})
    return response["messages"][-1].content
