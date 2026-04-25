"""Configuration for the Smart Study Assistant."""
import os

# API Key — set your Google API key here
os.environ.setdefault("GOOGLE_API_KEY", "your-api-key-here")

# Model settings
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/embedding-001"
LLM_TEMPERATURE = 0

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval settings
TOP_K = 3

# ChromaDB settings
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "study_notes"
