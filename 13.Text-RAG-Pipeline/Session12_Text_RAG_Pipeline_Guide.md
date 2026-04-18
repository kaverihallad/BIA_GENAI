# Session 12: Text-RAG Pipeline
## Building Your First Retrieval-Augmented Generation System

**Duration:** 3 hours | **Level:** Beginner | **Module:** Module 4 - Retrieval-Augmented Generation & Multimodal Systems

---

## 1. Quick Recap: Sessions 10-11 → Session 12

You've learned a lot already. Let's connect the dots:

| Session | Topic | You Learned | Output |
|---------|-------|-------------|--------|
| **10** | Embeddings | How text becomes vectors | Similarity scores between texts |
| **11** | Vector Databases | How to store & search vectors | FAISS, Chroma, hybrid search |
| **12** | Text-RAG Pipeline | How to build end-to-end RAG | Full pipeline: Load → Chunk → Retrieve → Generate |

**What you already know:**
- Text embeddings (Session 10) — converting text to dense vectors
- Similarity search — finding related documents using embeddings
- FAISS and Chroma — storing and querying vectors efficiently

**What's new in Session 12:**
- Loading real documents from files
- **Chunking** — breaking large documents into retrieval-friendly pieces
- Building a complete **retrieval chain** that feeds context to an LLM
- **Evaluating** RAG quality with metrics like Precision@k and F1@k
- The **router pattern** — deciding when to use retrieval vs. direct generation

---

## 2. What is RAG? (The Big Picture)

### The Problem

Large Language Models have limitations:
- **Knowledge cutoff:** Trained on data up to a certain date; don't know about recent events
- **Hallucinations:** Generate plausible-sounding but false information
- **No access to private data:** Can't answer questions about your company's documents, internal wikis, or proprietary datasets

### The Solution: Retrieval-Augmented Generation (RAG)

RAG solves this by combining retrieval with generation:

1. **Retrieve** relevant context from a knowledge base (your documents)
2. **Feed** that context to the LLM
3. **Generate** an answer grounded in real, retrieved information

### Visual: The RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────┘

Your Documents       │      OFFLINE (One-time setup)
    ↓               │
[LOAD]              │    Load documents into memory
    ↓               │
[CHUNK]             │    Split into retrieval-friendly chunks
    ↓               │
[EMBED]             │    Convert chunks to vectors
    ↓               │
[STORE]             │    Store vectors in vector database
    ↓               │
Vector Store        │
┌──────────────────────┤      ONLINE (Query time)
│                      │
User Query            │
    ↓                 │
[RETRIEVE]            │    Find top-k relevant chunks
    ↓                 │
Retrieved Context     │
    ↓                 │
[GENERATE]            │    LLM synthesizes answer
    ↓                 │
Answer                │
```

### Why RAG Beats Fine-Tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Setup time** | Hours | Days/weeks |
| **Update data** | Add documents anytime | Retrain entire model |
| **Traceability** | Can show source chunks | Black box |
| **Cost** | Cheaper | Expensive |
| **When to use** | Most real-world apps | When LLM behavior is the problem |

---

## 3. Step 1: Loading Documents

Before you can retrieve from documents, you need to load them.

### LangChain Document Loaders

LangChain provides loaders for many formats. For beginners, we'll focus on:

#### Loading Text Files

```python
from langchain.document_loaders import TextLoader

# Simple example: load a plain text file
loader = TextLoader("sample.txt")
docs = loader.load()

# docs is a list of Document objects
print(len(docs))  # number of documents
print(docs[0].page_content[:100])  # first 100 chars
print(docs[0].metadata)  # metadata (e.g., filename)
```

#### Understanding the Document Object

Each document has two key parts:

```python
from langchain.schema import Document

doc = Document(
    page_content="The quick brown fox...",
    metadata={"source": "sample.txt", "page": 1}
)
```

- **page_content:** The actual text
- **metadata:** Dictionary with source, page, or custom fields

#### Loading PDF Files (Brief Overview)

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample.pdf")
docs = loader.load()
# Each page becomes a separate document
```

### Practical Example: Load and Inspect

```python
# Load a sample document
from langchain.document_loaders import TextLoader

loader = TextLoader("faq.txt")
documents = loader.load()

print(f"Loaded {len(documents)} documents")
print(f"First document length: {len(documents[0].page_content)} characters")
print(f"Metadata: {documents[0].metadata}")
```

---

## 4. Step 2: Chunking Strategies

### Why Chunk?

Documents can be very long (thousands of words), but:
- LLMs have limited context windows (e.g., 4K, 8K tokens)
- Retrievers work better with focused chunks (not entire 100-page documents)
- Overlapping chunks help capture information that spans boundaries

### Strategy 1: Fixed-Size Chunking (CharacterTextSplitter)

Split by number of characters, with overlap.

**Pros:** Simple, fast
**Cons:** May split sentences in the middle

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,        # characters per chunk
    chunk_overlap=200       # characters of overlap
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")
```

### Strategy 2: Recursive Chunking (Recommended)

Splits recursively by separators: `["\n\n", "\n", " ", ""]`

This keeps paragraphs and sentences together, making chunks more coherent.

**Pros:** Preserves structure, better for semantic meaning
**Cons:** Slightly slower

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,             # tokens (approximately 4 chars per token)
    chunk_overlap=50,           # tokens
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")
print(f"Example chunk:\n{chunks[0].page_content}")
```

### Best Practice: Benchmark Settings

Research shows these work well for most use cases:

- **Chunk size:** 512 tokens (~2000 characters)
- **Overlap:** 50-100 tokens (10-20% of chunk size)
- **Expected accuracy:** ~69% baseline

This overlap improves retrieval by up to 9% because important information at chunk boundaries isn't lost.

### Visualization: How Chunking Works

```
Original Document:
┌────────────────────────────────────────────────────────────────┐
│ Paragraph 1: "The history of AI started in the 1950s. Alan     │
│ Turing asked: can machines think? This question led to the...  │
│                                                                 │
│ Paragraph 2: "In 1956, the Dartmouth Workshop formalized AI    │
│ as a field. Researchers were optimistic about building human-  │
│ level AI quickly. This optimism lasted until...                │
└────────────────────────────────────────────────────────────────┘

With Recursive Chunking (chunk_size=512, overlap=50):
┌─────────────────────────────────┐
│ Chunk 1:                         │
│ "The history of AI started      │
│  in the 1950s. Alan Turing...   │
│  ...led to the..."              │
├─────┬───────────────────────────┤
│  Ov │ Chunk 2:                  │
│ erl │ "...led to the...          │
│ ap  │  In 1956, the Dartmouth   │
│     │  Workshop formalized AI... │
│     │  ...This optimism lasted   │
│     │  until..."                 │
│     └────────────────────────────┘
│
└─ Overlap ensures we don't lose info at boundaries
```

### Comparing Strategies

```python
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

# Load your document
from langchain.document_loaders import TextLoader
loader = TextLoader("sample.txt")
docs = loader.load()

# Strategy 1: Fixed-size
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
char_chunks = char_splitter.split_documents(docs)

# Strategy 2: Recursive (recommended)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
recursive_chunks = recursive_splitter.split_documents(docs)

print(f"CharacterTextSplitter: {len(char_chunks)} chunks")
print(f"RecursiveCharacterTextSplitter: {len(recursive_chunks)} chunks")

# RecursiveCharacterTextSplitter usually creates more chunks (smaller, focused ones)
```

---

## 5. Step 3: Embed & Store (Recap from Session 11)

You already know this! Quick reminder:

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Step 1: Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Step 2: Create chunks (from previous step)
# chunks = [Document(...), Document(...), ...]

# Step 3: Store in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_data"
)

print(f"Stored {vectorstore._collection.count()} chunks in Chroma")
```

**Key point:** The vector store is now ready for retrieval!

---

## 6. Step 4: The Retriever

A **retriever** searches your vector store and returns the most relevant chunks.

### Basic Retriever

```python
# Convert vector store to retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Return top 3 most relevant chunks
)

# Test it with a query
query = "When did AI start?"
retrieved_docs = retriever.invoke(query)

print(f"Retrieved {len(retrieved_docs)} documents:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content[:200])  # First 200 chars
```

### Retriever Parameters

| Parameter | Effect |
|-----------|--------|
| **k** | Number of chunks to return (top-k) |
| **fetch_k** | Number of chunks to fetch before filtering |
| **lambda_mult** | For MMR: balance between relevance (0) and diversity (1) |

### Retriever Types

#### 1. Similarity Search (Default)

Returns chunks most similar to query by embedding distance.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

#### 2. Maximal Marginal Relevance (MMR)

Returns relevant chunks that are also diverse (not repetitive).

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)
# lambda_mult=1.0 → pure similarity
# lambda_mult=0.0 → pure diversity
```

### Testing Your Retriever

```python
# Create some test queries
test_queries = [
    "When did AI start?",
    "Who invented deep learning?",
    "What is a neural network?"
]

for query in test_queries:
    retrieved = retriever.invoke(query)
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(retrieved)} chunks:")
    for doc in retrieved:
        print(f"  - {doc.page_content[:80]}...")
```

---

## 7. Step 5: The Generator (RAG Chain)

Now combine the retriever with an LLM to **generate** answers grounded in retrieved context.

### The RAG Prompt Template

The prompt tells the LLM: "Here's context from our documents. Use it to answer the question."

```python
from langchain.prompts import PromptTemplate

rag_prompt = PromptTemplate.from_template(
    """You are a helpful assistant. Answer the following question based on the provided context.
    
Context:
{context}

Question: {question}

Answer:"""
)
```

### Building the RAG Chain

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# Step 1: LLM for generation
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Step 2: Document chain (combines retrieved docs into context)
document_chain = create_stuff_documents_chain(llm, rag_prompt)

# Step 3: Full RAG chain (retriever + document chain)
rag_chain = create_retrieval_chain(retriever, document_chain)

# Step 4: Query!
response = rag_chain.invoke({"input": "When did AI start?"})
print(response["answer"])
```

### What Happens Inside the Chain

```
User Query: "When did AI start?"
    ↓
[RETRIEVER] → Finds top 3 relevant chunks
    ↓
Chunks:
  1. "AI research began in 1956 at the Dartmouth Conference..."
  2. "The term 'artificial intelligence' was coined by John McCarthy in 1956..."
  3. "Before 1956, there was work on automated reasoning..."
    ↓
[CONTEXT FORMATTER] → Creates context string:
  "1. AI research began in 1956 at the Dartmouth Conference...
   2. The term 'artificial intelligence' was coined by John McCarthy in 1956...
   3. Before 1956, there was work on automated reasoning..."
    ↓
[LLM] → Receives prompt:
  "Context: [above]
   Question: When did AI start?
   Answer:"
    ↓
[GENERATION] → "AI as a formal field started in 1956 with the Dartmouth Conference,
                 though earlier work on automated reasoning existed."
```

### Complete Example: Simple FAQ Bot

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 1. Load documents
loader = TextLoader("faq.txt")
documents = loader.load()

# 2. Chunk documents
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 3. Embed and store
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Create RAG chain
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
rag_prompt = PromptTemplate.from_template(
    """Answer based on this context:
{context}

Question: {question}

Answer:"""
)
document_chain = create_stuff_documents_chain(llm, rag_prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# 6. Query!
result = rag_chain.invoke({"input": "What is your return policy?"})
print(result["answer"])
```

---

## 8. The Router Pattern

Not every query needs retrieval. Some questions are general knowledge.

### Why Route?

- **Cost:** API calls to retrieve documents cost money
- **Speed:** Direct LLM call is faster than retriever + LLM
- **Quality:** Some questions (e.g., "What is 2+2?") don't need retrieved context

### Simple Router: Query Classification

Route based on whether the query looks like it needs specific document knowledge:

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Create a router prompt
router_prompt = PromptTemplate.from_template(
    """Given a user query, determine if it requires document retrieval or can be answered
    directly by an LLM.

Query: {question}

Respond with only "RETRIEVE" or "DIRECT"."""
)

router_chain = router_prompt | llm

# Test
test_queries = [
    "What is photosynthesis?",           # DIRECT (general knowledge)
    "What's our return policy?",         # RETRIEVE (specific to docs)
    "How many countries are there?",     # DIRECT (general knowledge)
    "Who is our CEO?",                   # RETRIEVE (specific to company)
]

for query in test_queries:
    route = router_chain.invoke({"question": query})
    print(f"Query: {query}")
    print(f"Route: {route.content.strip()}\n")
```

### Full Routing Logic

```python
def answer_question(question, retriever, llm):
    """Route to retrieval or direct LLM based on query."""
    
    # 1. Classify the query
    router_prompt = PromptTemplate.from_template(
        """Is this query specific to certain documents, or general knowledge?
        
Query: {question}

Answer with only "RETRIEVE" or "DIRECT"."""
    )
    router = router_prompt | llm
    route_result = router.invoke({"question": question})
    route = route_result.content.strip()
    
    if route == "RETRIEVE":
        # Use RAG
        print(f"→ Using retrieval for: {question}")
        # (use rag_chain from previous section)
        return rag_chain.invoke({"input": question})
    else:
        # Direct LLM
        print(f"→ Using direct LLM for: {question}")
        response = llm.invoke(question)
        return {"answer": response.content}

# Test it
result = answer_question("What is AI?", retriever, llm)
print(result["answer"])
```

---

## 9. Evaluating Your RAG Pipeline

How do you know if your RAG system works well?

### Why Evaluate?

- Without metrics, you're guessing if retrieval is actually good
- Different chunking strategies, retriever types, and LLMs need comparison
- Metrics help debug problems (is retrieval bad, or generation, or both?)

### Metric 1: Precision@k

**Definition:** Of the top-k retrieved documents, how many are relevant?

$$\text{Precision@k} = \frac{\text{# relevant docs in top-k}}{\text{k}}$$

**Example:**
- Retrieved 5 docs, 3 are relevant → Precision@5 = 3/5 = 0.6 (60%)
- Retrieved 10 docs, 8 are relevant → Precision@10 = 8/10 = 0.8 (80%)

**Code:**

```python
def precision_at_k(retrieved_docs, relevant_doc_ids, k=5):
    """Calculate precision@k."""
    # Only look at top k
    top_k = retrieved_docs[:k]
    
    # Count how many are relevant
    relevant_count = sum(1 for doc in top_k if doc.metadata.get("id") in relevant_doc_ids)
    
    return relevant_count / k

# Example
relevant = {"doc_1", "doc_3", "doc_7"}  # We know these are relevant
retrieved = [
    {"metadata": {"id": "doc_1"}},  # relevant ✓
    {"metadata": {"id": "doc_2"}},  # not relevant ✗
    {"metadata": {"id": "doc_3"}},  # relevant ✓
    {"metadata": {"id": "doc_4"}},  # not relevant ✗
    {"metadata": {"id": "doc_5"}},  # not relevant ✗
]

p_at_5 = precision_at_k(retrieved, relevant, k=5)
print(f"Precision@5 = {p_at_5:.2f}")  # 2/5 = 0.40
```

### Metric 2: Recall@k

**Definition:** Of all relevant documents, what fraction is in the top-k?

$$\text{Recall@k} = \frac{\text{# relevant docs in top-k}}{\text{total # relevant docs}}$$

**Example:**
- 10 relevant docs exist, 6 are in top-5 → Recall@5 = 6/10 = 0.6 (60%)
- 10 relevant docs exist, 9 are in top-10 → Recall@10 = 9/10 = 0.9 (90%)

**Code:**

```python
def recall_at_k(retrieved_docs, relevant_doc_ids, k=5):
    """Calculate recall@k."""
    top_k = retrieved_docs[:k]
    relevant_count = sum(1 for doc in top_k if doc.metadata.get("id") in relevant_doc_ids)
    
    # Total number of relevant docs
    total_relevant = len(relevant_doc_ids)
    
    if total_relevant == 0:
        return 1.0  # If no relevant docs, recall is perfect (edge case)
    
    return relevant_count / total_relevant

# Example (same as before)
r_at_5 = recall_at_k(retrieved, relevant, k=5)
print(f"Recall@5 = {r_at_5:.2f}")  # 2/3 ≈ 0.67
```

### Metric 3: F1@k

**Definition:** Harmonic mean of precision and recall. Balances both metrics.

$$\text{F1@k} = 2 \cdot \frac{\text{Precision@k} \times \text{Recall@k}}{\text{Precision@k} + \text{Recall@k}}$$

**Code:**

```python
def f1_at_k(retrieved_docs, relevant_doc_ids, k=5):
    """Calculate F1@k."""
    precision = precision_at_k(retrieved_docs, relevant_doc_ids, k)
    recall = recall_at_k(retrieved_docs, relevant_doc_ids, k)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Example
f1 = f1_at_k(retrieved, relevant, k=5)
print(f"F1@5 = {f1:.2f}")
```

### Simple Evaluation Workflow

```python
# Step 1: Create test cases (question + relevant doc IDs)
test_cases = [
    {
        "question": "When did AI start?",
        "relevant_docs": {"doc_1", "doc_5"}  # Docs we know answer this
    },
    {
        "question": "Who invented neural networks?",
        "relevant_docs": {"doc_3", "doc_8"}
    }
]

# Step 2: For each test case, retrieve and evaluate
results = []
for test in test_cases:
    retrieved = retriever.invoke(test["question"])
    retrieved_ids = [doc.metadata.get("id") for doc in retrieved]
    
    p = precision_at_k(retrieved, test["relevant_docs"], k=3)
    r = recall_at_k(retrieved, test["relevant_docs"], k=3)
    f = f1_at_k(retrieved, test["relevant_docs"], k=3)
    
    results.append({"question": test["question"], "p@3": p, "r@3": r, "f1@3": f})
    print(f"Q: {test['question']}")
    print(f"  P@3={p:.2f}, R@3={r:.2f}, F1@3={f:.2f}\n")

# Step 3: Average metrics
avg_p = sum(r["p@3"] for r in results) / len(results)
avg_r = sum(r["r@3"] for r in results) / len(results)
avg_f = sum(r["f1@3"] for r in results) / len(results)

print(f"Average: P@3={avg_p:.2f}, R@3={avg_r:.2f}, F1@3={avg_f:.2f}")
```

### Other Metrics (Brief Mention)

- **NDCG (Normalized Discounted Cumulative Gain):** Accounts for ranking position (top results matter more)
- **MRR (Mean Reciprocal Rank):** Average position of first relevant doc
- **Ragas:** Automated RAG evaluation using LLMs
- **DeepEval:** Framework for LLM evaluation
- **TruLens:** Monitoring for RAG systems

For beginners, focus on **Precision@k, Recall@k, and F1@k**.

---

## 10. Putting It All Together: Complete RAG System

Here's the complete pipeline from start to finish:

```python
"""
Complete RAG Pipeline: Load → Chunk → Embed → Store → Retrieve → Generate
"""

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ============ SETUP: One-time (offline) ============

print("Step 1: Loading documents...")
loader = TextLoader("documents.txt")
documents = loader.load()
print(f"  Loaded {len(documents)} documents")

print("\nStep 2: Chunking documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"  Created {len(chunks)} chunks")

print("\nStep 3: Embedding and storing...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
print(f"  Stored in Chroma")

print("\nStep 4: Creating retriever...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print(f"  Retriever ready (returning top-3 chunks)")

# ============ SETUP: One-time initialization ============

print("\nStep 5: Initializing LLM...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
print(f"  LLM ready")

print("\nStep 6: Creating RAG chain...")
rag_prompt = PromptTemplate.from_template(
    """You are a helpful assistant. Answer the question based on the following context.
    If the context doesn't contain the answer, say "I don't have this information."
    
Context:
{context}

Question: {question}

Answer:"""
)
document_chain = create_stuff_documents_chain(llm, rag_prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)
print(f"  RAG chain ready\n")

# ============ QUERY TIME: Interactive ============

print("=" * 60)
print("RAG SYSTEM READY. Enter queries (type 'quit' to exit):")
print("=" * 60)

while True:
    query = input("\n> ").strip()
    
    if query.lower() == 'quit':
        break
    
    if not query:
        continue
    
    # Run RAG chain
    result = rag_chain.invoke({"input": query})
    
    print(f"\nAnswer: {result['answer']}")
    print("\nContext used:")
    for i, doc in enumerate(result["context"], 1):
        print(f"  [{i}] {doc.page_content[:100]}...")
```

### Testing with Sample Documents

Create a sample document for testing:

```python
# Create sample_docs.txt with this content:
"""
The History of Artificial Intelligence

AI research began officially in 1956 at the Dartmouth Summer Research Project.
Key researchers included John McCarthy, Marvin Minsky, Claude Shannon, and Nathaniel
Rochester. This conference formalized artificial intelligence as an academic field.

Early Progress (1956-1974)
In the early years, researchers were optimistic. They believed human-level AI could
be achieved within 20 years. Early systems like ELIZA and SHRDLU showed promise in
specific domains.

AI Winter (1974-1980, 1987-1993)
As expectations weren't met, funding dried up and progress slowed. These periods
are called "AI winters" because research nearly froze.

Deep Learning Revolution (2012-Present)
The availability of large datasets and powerful GPUs enabled deep learning.
AlexNet won ImageNet in 2012. Since then, breakthroughs in NLP (transformers,
GPT, BERT) and computer vision have accelerated.
"""

# Then run the complete RAG system above with:
loader = TextLoader("sample_docs.txt")
```

### Sample Queries and Answers

```
> When did AI research start?
Answer: AI research began officially in 1956 at the Dartmouth Summer Research Project
with key researchers including John McCarthy, Marvin Minsky, Claude Shannon, and
Nathaniel Rochester.

> What is an AI winter?
Answer: AI winters refer to periods of reduced funding and slowed progress in AI
research. There were two main AI winters: 1974-1980 and 1987-1993. During these
periods, early AI systems failed to meet optimistic expectations, causing funding
to dry up.

> Who invented the transformer?
Answer: I don't have this information in the provided context.
```

---

## 11. Exercises

### Exercise 1: Build a FAQ Bot

**Objective:** Create a RAG system that answers FAQs.

**Steps:**

1. Create a file `faq.txt` with 5-10 Q&A pairs:
   ```
   Q: What is your refund policy?
   A: We offer 30-day refunds on all items in original condition.
   
   Q: How long does shipping take?
   A: Standard shipping takes 5-7 business days...
   ```

2. Load, chunk, embed, and store in Chroma

3. Create a RAG chain

4. Test with queries like:
   - "Can I return items?"
   - "How fast is shipping?"
   - "Do you offer international shipping?"

5. Evaluate with precision/recall metrics

**Expected:** System should retrieve relevant Q&A pairs and generate clear answers.

### Exercise 2: Experiment with Chunk Sizes

**Objective:** Compare how chunk size affects retrieval quality.

**Steps:**

1. Use the same documents from Exercise 1

2. Create three RAG systems with different chunk sizes:
   - System A: chunk_size=256, overlap=25
   - System B: chunk_size=512, overlap=50 (recommended)
   - System C: chunk_size=1024, overlap=100

3. Test each system with the same 5 queries

4. Evaluate with precision@3 and F1@3 metrics

5. Compare results

**Expected:** chunk_size=512 should perform best (not too large, not too small).

---

## 12. Quick Reference Card

### Installation

```bash
pip install langchain langchain-google-genai langchain-chroma chromadb
```

### Key Imports

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
```

### Complete Code Snippet (Minimal)

```python
# Load
from langchain.document_loaders import TextLoader
docs = TextLoader("file.txt").load()

# Chunk
from langchain.text_splitter import RecursiveCharacterTextSplitter
chunks = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50).split_documents(docs)

# Embed & Store
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(
    chunks,
    GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
)

# Retrieve
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Generate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = PromptTemplate.from_template("Answer: {context}\nQ: {question}")
chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# Query
result = chain.invoke({"input": "Your question"})
print(result["answer"])
```

### Common Parameters

| Component | Parameter | Example |
|-----------|-----------|---------|
| RecursiveCharacterTextSplitter | chunk_size | 512 |
| RecursiveCharacterTextSplitter | chunk_overlap | 50 |
| as_retriever() | k | 3 |
| as_retriever() | search_type | "similarity" or "mmr" |
| ChatGoogleGenerativeAI | model | "gemini-2.5-flash" |
| GoogleGenerativeAIEmbeddings | model | "models/gemini-embedding-001" |

### Troubleshooting

| Problem | Solution |
|---------|----------|
| ImportError: No module named 'langchain_chroma' | `pip install langchain-chroma` |
| "No API key found" | Set `GOOGLE_API_KEY` environment variable |
| Slow retrieval | Reduce chunk_size or use MMR retriever |
| Low precision | Increase chunk_overlap or adjust k |
| LLM hallucinating | Check if retrieved context actually answers query |

---

## Summary

In this session, you've learned:

1. **Loading:** How to load documents from files
2. **Chunking:** RecursiveCharacterTextSplitter with 512 tokens and 50 tokens overlap
3. **Storing:** Using Chroma to store embeddings (from Session 11)
4. **Retrieval:** Fetching top-k relevant chunks for a query
5. **Generation:** Using an LLM to synthesize answers from context
6. **Routing:** Deciding when retrieval is needed vs. direct LLM
7. **Evaluation:** Measuring quality with Precision@k, Recall@k, F1@k

**You now have everything needed to build production-ready RAG systems!**

Next session (Session 13), we'll extend RAG to multimodal systems (images, tables, etc.) and explore advanced evaluation techniques.

---

## Further Reading

- [LangChain Documentation](https://docs.langchain.com/)
- [Chroma Vector Database](https://docs.trychroma.com/)
- [Google Generative AI API](https://ai.google.dev/)
- [RAG Best Practices](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)

---

**Happy building!** 🚀
