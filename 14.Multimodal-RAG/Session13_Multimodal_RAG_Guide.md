# Session 13: Multimodal RAG
## Beyond Text — Building RAG Systems That See, Read, and Watch

**Duration:** 3 hours | **Level:** Beginner | **Location:** BIA Pune

---

## Table of Contents
1. [Quick Recap: Session 12 → Session 13](#1-quick-recap-session-12--session-13)
2. [Why Multimodal RAG?](#2-why-multimodal-rag)
3. [Gemini's Multimodal Superpowers](#3-geminis-multimodal-superpowers)
4. [Strategy 1: Describe-Then-Embed (Recommended for Beginners)](#4-strategy-1-describe-then-embed-recommended-for-beginners)
5. [Strategy 2: Native Multimodal Embeddings (Preview)](#5-strategy-2-native-multimodal-embeddings-preview)
6. [YouTube Video RAG](#6-youtube-video-rag)
7. [Document + Chart + YouTube Assistant](#7-document--chart--youtube-assistant)
8. [Practical Considerations](#8-practical-considerations)
9. [Exercises](#9-exercises)
10. [Quick Reference Card](#10-quick-reference-card)

---

## 1. Quick Recap: Session 12 → Session 13

In **Session 12**, you learned the Text-RAG Pipeline:
- **Chunking:** Breaking documents into manageable pieces
- **Embedding:** Converting text chunks to vectors
- **Retrieval:** Finding relevant chunks based on a query
- **Generation:** Using the retrieved chunks to answer questions

| Aspect | Session 12 (Text-RAG) | Session 13 (Multimodal-RAG) |
|--------|----------------------|--------------------------|
| **Input Types** | Text documents only | Text + Images + Videos |
| **Embedding Model** | Text embeddings (GoogleGenerativeAIEmbeddings) | Text OR Multimodal embeddings (Gemini Embedding 2) |
| **Knowledge Sources** | PDFs, markdown, HTML | PDFs + Charts + YouTube Videos |
| **LLM Capability** | Text-to-text | Vision-aware (Gemini can see images) |
| **Retrieval Strategy** | Text-based similarity | Text-based OR vision-based |

**What's New in Session 13:**
- Images and charts as structured knowledge
- Video transcripts as queryable content
- Multimodal embeddings to connect text queries to images
- Vision-powered LLM that can interpret visual content

---

## 2. Why Multimodal RAG?

Real-world knowledge exists in many forms. Consider:

**Medical Domain:**
- Text: Patient reports, medical literature
- Images: X-rays, CT scans, pathology slides
- Problem: A doctor asks "What does this X-ray show?" — how do you retrieve relevant similar cases and medical text together?

**Finance Domain:**
- Text: Annual reports, analyst notes
- Charts: Stock price trends, quarterly performance graphs
- Videos: Earnings call recordings
- Problem: "Show me companies with similar growth patterns" — you need to search across text reports AND chart images

**Education Domain:**
- Text: Lecture notes, textbooks
- Images: Diagrams, concept maps
- Videos: Full lecture recordings
- Problem: Students ask "Explain this concept" but it's best explained by a video or diagram

**Without Multimodal RAG:**
- You search only the text, miss visual insights
- Vision models can see the image but have no context from your knowledge base

**With Multimodal RAG:**
- Text queries find relevant images, charts, and videos
- LLM can reason over both text and visual content together
- You can answer questions like: "Find reports similar to this chart I'm showing you"

---

## 3. Gemini's Multimodal Superpowers

Gemini 2.5 Flash is **natively multimodal** — it can process text and images in a single API call.

### What This Means:
Instead of separate text and vision models, Gemini sees text, images, PDFs, and more all at once.

### Simple Example: Ask About an Image

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64

# Initialize Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Example 1: Image from URL
message = HumanMessage(
    content=[
        {"type": "text", "text": "What is this image showing? Describe it briefly."},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/chart.png"
            }
        }
    ]
)
response = llm.invoke(message)
print(response.content)
```

**Output:** Gemini describes the chart, identifies trends, explains what it shows — all in one call.

### Example 2: Local Image (Base64 Encoded)

```python
import base64

# Read and encode a local image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")

image_data = encode_image("sales_chart.png")

message = HumanMessage(
    content=[
        {"type": "text", "text": "Analyze this sales chart. What are the key trends?"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_data}"
            }
        }
    ]
)

response = llm.invoke(message)
print(response.content)
```

### Key Takeaway:
Gemini can understand images natively. This is the foundation of multimodal RAG.

---

## 4. Strategy 1: Describe-Then-Embed (Recommended for Beginners)

This is the simplest and most practical approach for building multimodal RAG right now.

### The Idea:
1. **For each image:** Use Gemini to write a detailed text description
2. **Embed the description:** Use standard text embeddings
3. **Store:** The description + reference to the original image
4. **At query time:** Retrieve by text similarity, then pass the original image to Gemini

### Why This Works:
- Text embeddings are mature, fast, and cheap
- Gemini can describe any visual content perfectly
- At runtime, the LLM can see the actual image (no quality loss)
- Works with existing RAG infrastructure

### Complete Code Example

```python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import base64
import os

# ============================================================================
# Step 1: Describe Images
# ============================================================================

def encode_image(image_path):
    """Convert image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def describe_image_with_gemini(image_path, llm):
    """Use Gemini to generate a detailed text description of an image."""
    image_data = encode_image(image_path)
    
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """Describe this image in detail for a knowledge base. 
                Include:
                1. What the image shows (chart, diagram, photo, etc.)
                2. Key data, numbers, or labels visible
                3. Main insights or conclusions
                4. Any text or annotations
                Keep it concise but informative (100-150 words)."""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }
            }
        ]
    )
    
    response = llm.invoke(message)
    return response.content


# ============================================================================
# Step 2: Build Multimodal RAG with Describe-Then-Embed
# ============================================================================

class MultimodalRAG:
    def __init__(self, collection_name="multimodal_rag"):
        """Initialize the multimodal RAG system."""
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_multimodal"
        )
        # Store image data separately (path or base64)
        self.image_registry = {}
    
    def add_image(self, image_path, metadata=None):
        """
        Add an image to the RAG system:
        1. Describe it with Gemini
        2. Embed the description
        3. Store reference to original image
        """
        print(f"Processing image: {image_path}")
        
        # Step 1: Generate description
        description = describe_image_with_gemini(image_path, self.llm)
        print(f"  Description: {description[:100]}...")
        
        # Step 2: Create document with description
        doc_id = f"image_{len(self.image_registry)}"
        doc = Document(
            page_content=description,
            metadata={
                "type": "image",
                "image_path": image_path,
                "image_id": doc_id,
                **(metadata or {})
            }
        )
        
        # Step 3: Add to vector store
        self.vector_store.add_documents([doc])
        
        # Step 4: Store image for later retrieval
        self.image_registry[doc_id] = {
            "path": image_path,
            "base64": encode_image(image_path),
            "description": description
        }
        
        print(f"  ✓ Stored with ID: {doc_id}\n")
    
    def add_text(self, text_content, source="unknown", metadata=None):
        """Add text documents to the RAG system."""
        doc = Document(
            page_content=text_content,
            metadata={
                "type": "text",
                "source": source,
                **(metadata or {})
            }
        )
        self.vector_store.add_documents([doc])
        print(f"✓ Added text document from: {source}")
    
    def retrieve(self, query, k=3):
        """Retrieve relevant documents and images."""
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def answer_with_images(self, query):
        """
        Answer a question using retrieved text and images.
        1. Retrieve relevant documents
        2. For each image, include the actual image in the context
        3. Ask Gemini to answer using both text and images
        """
        # Step 1: Retrieve
        retrieved_docs = self.retrieve(query)
        
        # Step 2: Prepare context
        text_context = []
        images_context = []
        
        for doc in retrieved_docs:
            if doc.metadata.get("type") == "image":
                image_id = doc.metadata.get("image_id")
                if image_id in self.image_registry:
                    image_info = self.image_registry[image_id]
                    images_context.append({
                        "base64": image_info["base64"],
                        "description": doc.page_content
                    })
                    text_context.append(f"[Image: {image_info['description']}]")
            else:
                text_context.append(doc.page_content)
        
        # Step 3: Build message for Gemini
        message_content = [
            {
                "type": "text",
                "text": f"""Answer the following question using the provided context:

CONTEXT:
{chr(10).join(text_context)}

QUESTION: {query}

Provide a clear, concise answer."""
            }
        ]
        
        # Add images to the message
        for img in images_context:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img['base64']}"
                }
            })
        
        # Step 4: Get answer from Gemini
        message = HumanMessage(content=message_content)
        response = self.llm.invoke(message)
        
        return {
            "answer": response.content,
            "retrieved_count": len(retrieved_docs),
            "images_count": len(images_context),
            "text_count": len(text_context) - len(images_context)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Initialize RAG system
    rag = MultimodalRAG()
    
    # Add sample images (create these yourself)
    # rag.add_image("sales_chart.png", metadata={"quarter": "Q1 2024"})
    # rag.add_image("product_comparison.png", metadata={"type": "comparison"})
    
    # Add text documents
    rag.add_text(
        """Our Q1 2024 results showed strong growth in the tech sector.
        Mobile applications grew 25% YoY, while desktop solutions grew 12%.
        Market share increased from 15% to 18% in our target demographic.""",
        source="Q1 2024 Report"
    )
    
    # Query the system
    query = "What were the main growth areas in Q1 2024?"
    result = rag.answer_with_images(query)
    
    print(f"Question: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Retrieved: {result['text_count']} texts, {result['images_count']} images")
```

### Key Points:
- **Simple:** Just text embeddings + Gemini descriptions
- **Flexible:** Works with any image type
- **Cost-effective:** Vision API calls only during indexing, not retrieval
- **Quality:** Gemini sees the actual image when answering (no loss of visual detail)

---

## 5. Strategy 2: Native Multimodal Embeddings (Preview)

As of March 2026, **Gemini Embedding 2** (preview) is the first natively multimodal embedding model.

### What's New:
- **Single Vector Space:** Text, images, video, audio, and PDFs all map to the same 3072-dimensional space
- **Direct Image Retrieval:** A text query can directly find relevant images (no need to describe them first)
- **Unified Retrieval:** No separate logic for different modalities
- **Pricing:** $0.20 per 1M tokens (competitive)

### How It Works:

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import base64

# Initialize Gemini Embedding 2 (when GA)
# Note: Still in preview. Check docs for availability.
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"  # Future model name (preview: "models/gemini-2-embed-preview")
)

# Embed text
text_embedding = embeddings.embed_query("What is machine learning?")
# Output: 3072-dimensional vector

# Embed image (when fully available)
image_data = base64.standard_b64encode(open("chart.png", "rb").read()).decode()
image_embedding = embeddings.embed_query(
    f"data:image/png;base64,{image_data}"
)
# Output: 3072-dimensional vector in the SAME space as text!

# Now you can search for images using text queries:
# similarity_score = cosine_similarity(text_embedding, image_embedding)
```

### Advantages:
- **No descriptions needed:** Images embed directly
- **Better matching:** Visual similarity in the same space
- **Simpler code:** One embedding model for everything

### When to Use:
- Once Gemini Embedding 2 is generally available (GA)
- When you need direct image-to-text retrieval
- For large-scale multimodal systems

### Current Status (March 2026):
- In public preview
- API access available; pricing may change
- Check [Google AI docs](https://ai.google.dev) for latest availability

**For this session, we recommend Strategy 1 (Describe-Then-Embed) as it's proven and widely available.**

---

## 6. YouTube Video RAG

Learn how to build RAG over YouTube videos by extracting and chunking transcripts.

### The Idea:
1. **Extract transcript** from a YouTube video (free, no API key needed for public videos)
2. **Chunk the transcript** (like text documents)
3. **Embed and store** chunks
4. **Retrieve and answer** questions about the video

### Installation:

```bash
pip install youtube-transcript-api
```

### Complete YouTube RAG Example

```python
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

# ============================================================================
# Step 1: Extract YouTube Transcript
# ============================================================================

def extract_video_id(youtube_url):
    """Extract video ID from YouTube URL."""
    # Handles: youtube.com/watch?v=ID, youtu.be/ID
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None


def get_youtube_transcript(youtube_url):
    """Extract transcript from a YouTube video."""
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError(f"Invalid YouTube URL: {youtube_url}")
    
    try:
        # Fetch transcript (automatically tries multiple languages)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine all transcript entries into one text
        full_transcript = " ".join([item["text"] for item in transcript_list])
        
        return full_transcript
    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {str(e)}")


# ============================================================================
# Step 2: YouTube RAG System
# ============================================================================

class YouTubeRAG:
    def __init__(self, collection_name="youtube_rag"):
        """Initialize YouTube RAG system."""
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_youtube"
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
    
    def add_video(self, youtube_url, video_title=None):
        """
        Add a YouTube video to the RAG system:
        1. Extract transcript
        2. Chunk it
        3. Embed and store
        """
        print(f"Fetching transcript from: {youtube_url}")
        
        # Step 1: Get transcript
        try:
            transcript = get_youtube_transcript(youtube_url)
        except Exception as e:
            print(f"Error: {e}")
            return False
        
        print(f"  Transcript length: {len(transcript)} characters")
        
        # Step 2: Chunk transcript
        chunks = self.splitter.split_text(transcript)
        print(f"  Created {len(chunks)} chunks")
        
        # Step 3: Create documents
        docs = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "type": "youtube_transcript",
                    "video_url": youtube_url,
                    "video_title": video_title or "Unknown",
                    "chunk_index": i
                }
            )
            docs.append(doc)
        
        # Step 4: Store in vector database
        self.vector_store.add_documents(docs)
        print(f"  ✓ Stored {len(docs)} chunks\n")
        return True
    
    def retrieve(self, query, k=3):
        """Retrieve relevant transcript chunks."""
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def answer(self, query):
        """Answer a question about YouTube video(s)."""
        # Step 1: Retrieve relevant chunks
        retrieved_docs = self.retrieve(query, k=5)
        
        # Step 2: Build context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Step 3: Get answer from Gemini
        prompt = f"""Based on the following transcript excerpts, answer the question.

TRANSCRIPT EXCERPTS:
{context}

QUESTION: {query}

Provide a clear, accurate answer based on what's in the transcript."""
        
        response = self.llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "sources": [
                {
                    "title": doc.metadata.get("video_title"),
                    "url": doc.metadata.get("video_url"),
                    "chunk": doc.metadata.get("chunk_index")
                }
                for doc in retrieved_docs
            ]
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    rag = YouTubeRAG()
    
    # Example: Add an educational video (use any public YouTube video)
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with real video
    # rag.add_video(video_url, "My Educational Video")
    
    # Query
    # result = rag.answer("What was the main topic discussed?")
    # print(f"Answer: {result['answer']}")
    # print(f"Sources: {result['sources']}")
```

### Key Points:
- **No API key required** for public videos (youtube-transcript-api is free)
- **Automatic language handling** (finds transcript in available languages)
- **Clean chunking** breaks transcripts into searchable segments
- **Full retrieval** include timestamp metadata for reference

---

## 7. Document + Chart + YouTube Assistant

Combine all three modalities into a single intelligent assistant.

### Architecture:
```
Query
  ↓
[Router: Classify query type]
  ├→ Text question? → Search documents
  ├→ Visual question? → Search images/charts
  └→ Video question? → Search YouTube transcripts
  ↓
[Retrieval]
  ├→ Text docs
  ├→ Images/Charts (with descriptions)
  └→ Video transcripts
  ↓
[Gemini - Multimodal Answer]
  (Sees text, images, and transcript context)
```

### Complete Implementation

```python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
import base64
import re

class UnifiedMultimodalAssistant:
    """
    A single assistant that handles:
    - Text documents (PDFs, markdown, web pages)
    - Images/Charts (with Gemini descriptions)
    - YouTube videos (transcripts)
    """
    
    def __init__(self, collection_name="unified_rag"):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_unified"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        self.image_registry = {}
    
    # ========================================================================
    # Method 1: Add Text Documents
    # ========================================================================
    
    def add_document(self, content, source, doc_type="document"):
        """Add a text document."""
        chunks = self.text_splitter.split_text(content)
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "type": "text",
                    "source": source,
                    "doc_type": doc_type
                }
            )
            for chunk in chunks
        ]
        self.vector_store.add_documents(docs)
        print(f"✓ Added document '{source}' ({len(docs)} chunks)")
    
    # ========================================================================
    # Method 2: Add Images/Charts
    # ========================================================================
    
    def add_image(self, image_path, title=""):
        """Add an image or chart."""
        # Describe image with Gemini
        image_data = base64.standard_b64encode(
            open(image_path, "rb").read()
        ).decode()
        
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """Describe this image for a knowledge base. 
                    Include what it shows, key data/numbers, and main insights."""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                }
            ]
        )
        
        description = self.llm.invoke(message).content
        
        # Store
        image_id = f"image_{len(self.image_registry)}"
        doc = Document(
            page_content=description,
            metadata={
                "type": "image",
                "source": title or image_path,
                "image_id": image_id
            }
        )
        self.vector_store.add_documents([doc])
        self.image_registry[image_id] = {
            "path": image_path,
            "base64": image_data,
            "description": description
        }
        print(f"✓ Added image '{title or image_path}'")
    
    # ========================================================================
    # Method 3: Add YouTube Videos
    # ========================================================================
    
    def add_youtube_video(self, youtube_url, title=""):
        """Add a YouTube video transcript."""
        # Extract video ID
        match = re.search(
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
            youtube_url
        )
        if not match:
            print(f"Invalid YouTube URL: {youtube_url}")
            return
        
        video_id = match.group(1)
        
        try:
            # Get transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            full_transcript = " ".join([item["text"] for item in transcript_list])
        except Exception as e:
            print(f"Failed to fetch transcript: {e}")
            return
        
        # Chunk and store
        chunks = self.text_splitter.split_text(full_transcript)
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "type": "youtube",
                    "source": title or "YouTube Video",
                    "video_url": youtube_url
                }
            )
            for chunk in chunks
        ]
        self.vector_store.add_documents(docs)
        print(f"✓ Added YouTube video '{title or youtube_url}' ({len(docs)} chunks)")
    
    # ========================================================================
    # Unified Retrieval & Answering
    # ========================================================================
    
    def retrieve(self, query, k=5):
        """Retrieve from all modalities."""
        return self.vector_store.similarity_search(query, k=k)
    
    def answer(self, query):
        """
        Answer a question across all modalities.
        """
        # Step 1: Retrieve
        retrieved_docs = self.retrieve(query)
        
        # Step 2: Separate by type
        text_content = []
        image_ids = set()
        images_to_include = []
        
        for doc in retrieved_docs:
            doc_type = doc.metadata.get("type")
            
            if doc_type == "text":
                text_content.append(f"[{doc.metadata.get('source')}] {doc.page_content}")
            elif doc_type == "image":
                image_id = doc.metadata.get("image_id")
                if image_id not in image_ids and image_id in self.image_registry:
                    image_ids.add(image_id)
                    img_info = self.image_registry[image_id]
                    images_to_include.append(img_info)
                    text_content.append(f"[Image: {doc.metadata.get('source')}] {doc.page_content}")
            elif doc_type == "youtube":
                text_content.append(f"[YouTube: {doc.metadata.get('source')}] {doc.page_content}")
        
        # Step 3: Build message for Gemini
        message_content = [
            {
                "type": "text",
                "text": f"""Answer the following question using the provided context.

CONTEXT:
{chr(10).join(text_content)}

QUESTION: {query}

Provide a comprehensive answer that synthesizes information from documents, 
images, and videos."""
            }
        ]
        
        # Add images
        for img_info in images_to_include:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_info['base64']}"}
            })
        
        # Step 4: Get answer
        message = HumanMessage(content=message_content)
        response = self.llm.invoke(message)
        
        return {
            "answer": response.content,
            "stats": {
                "text_docs": len([d for d in retrieved_docs if d.metadata.get("type") == "text"]),
                "images": len(images_to_include),
                "youtube": len([d for d in retrieved_docs if d.metadata.get("type") == "youtube"])
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    assistant = UnifiedMultimodalAssistant()
    
    # Example 1: Add text document
    assistant.add_document(
        """Our company analysis shows:
        - Q1 Revenue: $2.5M (up 30% YoY)
        - Market growth in AI: +45%
        - Customer retention: 92%""",
        source="Q1 Report"
    )
    
    # Example 2: Add chart image
    # assistant.add_image("growth_chart.png", title="Revenue Growth Chart")
    
    # Example 3: Add YouTube video
    # assistant.add_youtube_video(
    #     "https://www.youtube.com/watch?v=...",
    #     title="AI Trends Discussion"
    # )
    
    # Query
    query = "What were the key results and growth areas?"
    result = assistant.answer(query)
    
    print(f"\nQuestion: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Retrieved: {result['stats']}")
```

### Usage Pattern:
```python
# Initialize once
assistant = UnifiedMultimodalAssistant()

# Add your knowledge sources
assistant.add_document(my_report, "Annual Report")
assistant.add_image("revenue_chart.png", "Revenue Growth")
assistant.add_youtube_video("https://youtube.com/...", "Earnings Call")

# Query across everything
answer = assistant.answer("What's the revenue outlook for next year?")
```

---

## 8. Practical Considerations

### Image Processing

**File Size Limits:**
- Gemini can handle images up to ~20MB
- For best performance, keep images under 2-3MB
- Compress before uploading if needed

**Format Support:**
- JPEG, PNG, WebP, GIF (animated)
- PDF (single or multi-page)

**Base64 Encoding:**
```python
import base64

# Check file size before encoding
import os
file_size_mb = os.path.getsize("chart.png") / 1024 / 1024
print(f"File size: {file_size_mb:.2f} MB")

# Encode
with open("chart.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
```

### Cost Optimization

**Vision API Calls:**
- Indexing: Expensive (Gemini describes each image)
- Retrieval: Cheap (only text embeddings)
- Answering: Moderate (multimodal LLM call)

**Strategy:**
- Describe images once during indexing
- Cache descriptions in vector store
- Retrieve images by description (cheap)
- Include actual images in final LLM call (unavoidable)

**Example Cost Estimate (100 images, 1000 queries):**
```
Indexing:  100 images × (vision call) ≈ $5-10
Retrieval: 1000 queries × (text embedding) ≈ $0.01
Answering: 1000 × (multimodal LLM) ≈ $10-20
Total: ~$15-30
```

### Performance Tips

1. **Chunk Size:** Keep video transcripts and documents at 300-500 words per chunk
2. **Retrieval Count:** Retrieve k=3-5 documents, not too many or context gets noisy
3. **Caching:** Cache images locally in base64 to avoid repeated downloads
4. **Batch Processing:** If adding many documents/images, batch them

### When to Use Each Strategy

| Scenario | Strategy | Why |
|----------|----------|-----|
| **Quick prototype** | Describe-Then-Embed | Simple, proven, fast to build |
| **Large image corpus** | Describe-Then-Embed | Cheaper at scale |
| **Direct image search** | Multimodal Embedding 2 | Once GA, cleaner semantics |
| **Video-heavy knowledge** | YouTube RAG | Transcripts are huge knowledge source |
| **Mixed content** | Unified Assistant | Videos + images + text together |

---

## 9. Exercises

### Exercise 1: YouTube Q&A Bot (30 minutes)

**Objective:** Build a system that answers questions about any YouTube video.

**Steps:**
1. Choose an educational YouTube video (15-30 minutes long)
2. Extract its transcript using youtube-transcript-api
3. Build a RAG system with ChromaDB
4. Ask 3-5 questions about the video's content
5. Verify answers are accurate

**Example Questions:**
- "What were the main points discussed?"
- "Who spoke and what were their roles?"
- "What examples or case studies were mentioned?"

**Starter Code:**
```python
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# TODO: Implement YouTubeRAG from Section 6
# TODO: Add a video
# TODO: Ask questions
```

**Success Criteria:**
- Transcript extracted successfully
- 5+ relevant chunks created
- Answers reference specific content from the video
- 3+ queries answered correctly

---

### Exercise 2: Image-Aware Chart Q&A (45 minutes)

**Objective:** Build a system that answers questions about financial or business charts.

**Steps:**
1. Find or create 3-4 simple chart images (bar charts, line graphs, pie charts)
   - Use tools like Excel, Google Sheets, or draw.io
   - Topics: sales, growth, market share, etc.
2. Implement the Describe-Then-Embed strategy from Section 4
3. Add the charts to your RAG system
4. Add supporting text (e.g., business context, report excerpts)
5. Query the system with questions about the data

**Example Charts & Questions:**
```
Chart 1: "Quarterly Revenue 2024"
Q: "Which quarter had the highest revenue?"
Q: "What's the trend from Q1 to Q4?"

Chart 2: "Market Share by Competitor"
Q: "Which company has the largest market share?"
Q: "How did our market share change?"
```

**Starter Code:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# TODO: Implement MultimodalRAG from Section 4
# TODO: Add 3-4 chart images
# TODO: Add supporting text documents
# TODO: Ask questions about charts + text
```

**Success Criteria:**
- All 3-4 images processed and stored
- Descriptions are accurate and detailed
- Queries return relevant images
- Answers combine text context + visual insights
- System correctly interprets chart data

---

### Exercise 3: Unified Knowledge System (60 minutes, Optional)

**Objective:** Build the complete UnifiedMultimodalAssistant from Section 7.

**Steps:**
1. Create a mini knowledge base:
   - 2-3 text documents (business reports, articles)
   - 2-3 chart images (Excel exports or screenshots)
   - 1 YouTube video (any educational topic)
2. Implement UnifiedMultimodalAssistant
3. Add all sources to the system
4. Ask 5+ questions that require combining different modalities

**Example Queries:**
- "Compare the growth shown in the chart to the text report"
- "What are the key takeaways across documents, charts, and videos?"
- "Summarize the analysis from all three sources"

**Success Criteria:**
- All sources added without errors
- Retrieval works across modalities
- Answers integrate information from multiple sources
- System correctly cites sources in answers

---

## 10. Quick Reference Card

### Installation
```bash
pip install langchain-google-genai langchain-chroma youtube-transcript-api
```

### Initialize Models

```python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# LLM (multimodal - can see text & images)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Text Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Multimodal Embeddings (preview, when available)
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
```

### Extract YouTube Transcript
```python
from youtube_transcript_api import YouTubeTranscriptApi
import re

def get_video_id(url):
    return re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)', url).group(1)

transcript = YouTubeTranscriptApi.get_transcript(get_video_id(url))
text = " ".join([item["text"] for item in transcript])
```

### Encode Image to Base64
```python
import base64

def encode_image(path):
    return base64.b64encode(open(path, "rb").read()).decode()

b64 = encode_image("chart.png")
```

### Describe Image with Gemini
```python
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe this image in detail."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    ]
)
description = llm.invoke(message).content
```

### Store in ChromaDB
```python
from langchain_chroma import Chroma
from langchain_core.documents import Document

vector_store = Chroma(
    collection_name="my_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

docs = [Document(page_content="...", metadata={"type": "image"})]
vector_store.add_documents(docs)
```

### Retrieve Documents
```python
results = vector_store.similarity_search("query", k=3)
for doc in results:
    print(doc.page_content)
    print(doc.metadata)
```

### Chunk Text
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_text(text)
```

### Multimodal LLM Call
```python
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "Question?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
)
response = llm.invoke(message)
```

---

## Summary

**What You Learned:**
1. ✓ Multimodal RAG extends RAG beyond text to images, charts, and videos
2. ✓ Strategy 1: Describe-Then-Embed is simple and practical (recommended for now)
3. ✓ Strategy 2: Multimodal Embeddings (Gemini Embedding 2) is the future
4. ✓ YouTube RAG lets you build Q&A over any video transcript
5. ✓ UnifiedMultimodalAssistant combines text, images, and videos seamlessly
6. ✓ Gemini 2.5 Flash is the foundation — it can "see" and reason over images

**Key Takeaway:**
Real-world knowledge is multimodal. By combining text RAG with image retrieval and video transcripts, you build intelligent systems that match how humans actually learn and make decisions.

---

**Next Steps:**
- Experiment with your own images and YouTube videos
- Try the exercises above
- Explore Gemini Embedding 2 once it's generally available
- Build production multimodal RAG systems with your own data

**Resources:**
- [Google AI Documentation](https://ai.google.dev)
- [LangChain Documentation](https://python.langchain.com)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [youtube-transcript-api](https://github.com/jderose9/youtube-transcript-api)

---

**Session 13 Complete!**

You've now learned to build RAG systems that see images, watch videos, and read text — just like humans do.

