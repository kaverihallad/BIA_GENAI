"""Build the RAG retrieval chain."""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import LLM_MODEL, LLM_TEMPERATURE, TOP_K


def get_llm():
    """Initialize the LLM."""
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore):
    """Build a RAG chain: retriever -> prompt -> LLM -> parser."""
    llm = get_llm()

    # TODO 7: Create a retriever from the vectorstore with search_kwargs={"k": TOP_K}
    retriever = None  # Replace this

    # TODO 8: Create a RAG prompt template that includes {context} and {question}
    # The prompt should tell the LLM to answer based ONLY on the provided context
    rag_prompt = None  # Replace this

    # TODO 9: Build the LCEL chain:
    # {"context": retriever | format_docs, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser()
    chain = None  # Replace this

    return chain


def ask_question(chain, question: str) -> str:
    """Ask a question using the RAG chain."""
    return chain.invoke(question)
