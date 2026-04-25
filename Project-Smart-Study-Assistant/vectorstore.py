"""Create and manage the vector store."""
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR, COLLECTION_NAME


def get_embeddings():
    """Initialize the embedding model."""
    # TODO 4: Return a GoogleGenerativeAIEmbeddings instance with EMBEDDING_MODEL
    pass


def create_vectorstore(chunks: list[str]):
    """Create a ChromaDB vector store from text chunks."""
    embeddings = get_embeddings()
    # TODO 5: Create a Chroma vectorstore from the chunks using Chroma.from_texts()
    # Include persist_directory=CHROMA_PERSIST_DIR
    # Hint: Chroma.from_texts(chunks, embeddings, persist_directory=CHROMA_PERSIST_DIR)
    pass


def load_vectorstore():
    """Load an existing ChromaDB vector store."""
    embeddings = get_embeddings()
    # TODO 6: Load existing Chroma from persist_directory
    # Hint: Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    pass
