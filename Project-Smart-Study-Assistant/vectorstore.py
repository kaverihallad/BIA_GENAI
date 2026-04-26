"""Create and manage the vector store."""
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR, COLLECTION_NAME


def get_embeddings():
    """Initialize the embedding model."""
    # TODO 4: Return a GoogleGenerativeAIEmbeddings instance with EMBEDDING_MODEL
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def create_vectorstore(chunks: list[str]):
    """Create a ChromaDB vector store from text chunks."""
    embeddings = get_embeddings()
    # Delete existing collection in-place (avoids OS-level file lock issues)
    if os.path.exists(CHROMA_PERSIST_DIR):
        try:
            existing = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME
            )
            existing.delete_collection()
        except Exception:
            pass  # Collection doesn't exist yet — that's fine
    return Chroma.from_texts(
        chunks,
        embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )


def load_vectorstore():
    """Load an existing ChromaDB vector store."""
    embeddings = get_embeddings()
    # TODO 6: Load existing Chroma from persist_directory
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
