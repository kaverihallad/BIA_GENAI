"""Load and chunk study documents."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_text_file(file_path: str) -> str:
    """Load a text file and return its contents."""
    # TODO 1: Read the file and return its content as a string
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def chunk_text(text: str) -> list[str]:
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    # TODO 2: Create a RecursiveCharacterTextSplitter with CHUNK_SIZE and CHUNK_OVERLAP
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    # TODO 3: Split the text into chunks and return them
    return splitter.split_text(text)


def load_and_chunk(file_path: str) -> list[str]:
    """Load a file and return chunks."""
    text = load_text_file(file_path)
    chunks = chunk_text(text)
    print(f"Loaded {file_path}: {len(text)} chars -> {len(chunks)} chunks")
    return chunks
