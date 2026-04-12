# Session 11: Vector DB Landscape
## Choosing the Right Vector Database for Your AI Application

**Module:** 4 - Retrieval-Augmented Generation & Multimodal Systems  
**Duration:** 3 hours  
**Level:** BEGINNER (BIA Pune)  
**Continuation from:** Session 10 (Vector Search 101 — embeddings, similarity, FAISS)

---

## 1. Quick Recap: From Session 10 to Session 11

### Journey Map

| Aspect | Session 10 (FAISS) | Session 11 (Production VectorDBs) |
|--------|-------------------|-----------------------------------|
| **Focus** | Understanding embeddings & similarity | Choosing & deploying the right tool |
| **Storage** | In-memory or local file | Persistent database |
| **Scale** | KB-MB range | MB to billions of vectors |
| **Filtering** | Manual post-processing | Native metadata filtering |
| **API** | Python-only | REST/GraphQL APIs |
| **Use Case** | Learning, experimentation | Production applications |

### Why This Progression Matters

In Session 10, you learned:
- **Embeddings**: Converting text to numbers (768-dimensional vectors with Gemini)
- **Similarity search**: Finding closest vectors using distance metrics
- **FAISS**: Building an in-memory index

Now in Session 11, we ask: *"What happens when you have 1 million documents? When you need to serve 100 queries per second? When you need to filter by document category while searching?"*

This is where specialized **Vector Databases** come in.

---

## 2. Why Move Beyond FAISS?

### FAISS Limitations

FAISS is brilliant for learning, but it's not production-grade:

| Limitation | Impact |
|-----------|--------|
| **No persistence** | Lose index when Python script exits |
| **No metadata filtering** | Can't search "show me papers about AI published after 2023" |
| **No built-in API** | Only Python; no way to query from mobile/web apps |
| **Single-machine** | Can't scale beyond RAM on one computer |
| **No update mechanism** | Rebuilding index is slow |

### What Production Vector DBs Add

```
┌─────────────────────────────────────────────────────┐
│         Production Vector Database Features          │
├─────────────────────────────────────────────────────┤
│ ✓ Persistence: Saves data to disk/cloud             │
│ ✓ Filtering: WHERE clauses on metadata              │
│ ✓ APIs: REST/GraphQL for any language               │
│ ✓ Scaling: Distributed across multiple servers      │
│ ✓ Updates: Add/delete/modify vectors without rebuild│
│ ✓ Hybrid Search: Keyword + semantic search          │
│ ✓ Monitoring: Latency, throughput metrics           │
└─────────────────────────────────────────────────────┘
```

### The 3 Categories

1. **Embedded Databases (Chroma)**
   - Runs inside your Python process
   - No server to manage
   - Perfect for learning and prototyping
   - Limited to single machine

2. **Self-Hosted (Weaviate)**
   - Run on your own servers
   - Full control, open-source
   - Moderate ops overhead
   - Hybrid search out-of-the-box

3. **Managed Cloud (Pinecone)**
   - Pay someone else to run it
   - Zero infrastructure
   - Scales instantly
   - Higher cost but zero headaches

---

## 3. ChromaDB — The Beginner-Friendly Choice

### What It Is

**Chroma** is an open-source embedded vector database. It runs inside your Python process—no servers, no Docker, no DevOps.

Think of it as: *"SQLite for vector search"*

### Installation

```bash
pip install chromadb google-generativeai langchain-google-generativeai
```

### Basic Example: Without LangChain

```python
import chromadb
from langchain_google_generativeai import GoogleGenerativeAIEmbeddings

# Initialize Chroma client (creates local database)
client = chromadb.Client()

# Create a collection (like a table)
collection = client.create_collection(name="documents")

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Add documents
documents = [
    "Python is a programming language used for AI development",
    "Machine learning models learn patterns from data",
    "Vector databases store embeddings for fast retrieval",
    "LangChain helps build LLM applications",
    "Embeddings convert text into numbers",
]

# Embed and add to Chroma
for i, doc in enumerate(documents):
    embedding = embeddings.embed_query(doc)
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[doc],
        metadatas=[{"source": "tutorial", "doc_num": i}]
    )

print(f"Added {len(documents)} documents to Chroma")

# Query: Find similar documents
query = "What is machine learning?"
query_embedding = embeddings.embed_query(query)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\n=== Search Results ===")
for i, doc in enumerate(results['documents'][0]):
    print(f"{i+1}. {doc}")
```

**Output:**
```
Added 5 documents to Chroma

=== Search Results ===
1. Machine learning models learn patterns from data
2. Vector databases store embeddings for fast retrieval
```

### With LangChain: Cleaner Syntax

```python
from langchain.vectorstores import Chroma
from langchain_google_generativeai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Prepare documents
docs = [
    Document(page_content="Python is a programming language used for AI development", 
             metadata={"source": "tutorial"}),
    Document(page_content="Machine learning models learn patterns from data",
             metadata={"source": "tutorial"}),
    Document(page_content="Vector databases store embeddings for fast retrieval",
             metadata={"source": "tutorial"}),
    Document(page_content="LangChain helps build LLM applications",
             metadata={"source": "tutorial"}),
    Document(page_content="Embeddings convert text into numbers",
             metadata={"source": "tutorial"}),
]

# Create Chroma vector store
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Save to disk
)

# Similarity search
results = vector_store.similarity_search("What is machine learning?", k=2)
print("=== Chroma Search Results ===")
for result in results:
    print(f"- {result.page_content}")

# Metadata filtering
results_filtered = vector_store.similarity_search(
    "embeddings",
    k=5,
    filter={"source": "tutorial"}
)
print(f"\n=== Filtered Results ({len(results_filtered)} found) ===")
for result in results_filtered:
    print(f"- {result.page_content}")
```

### When to Use Chroma

✅ **Perfect for:**
- Learning vector search
- Prototyping RAG applications
- Small datasets (< 1M vectors)
- Local development
- Embedded applications
- Quick POC (proof of concept)

❌ **Not suitable for:**
- Multi-server distributed systems
- Real-time updates with millions of vectors
- High query throughput (100+ qps)

### Chroma Limitations

| Limitation | Workaround |
|-----------|-----------|
| Single machine only | Use Pinecone/Weaviate for scale |
| Slower with 1M+ vectors | For large scale, switch VectorDB |
| Manual server management for production | Use managed Chroma Cloud (paid) |

---

## 4. Weaviate — The Hybrid Search Champion

### What It Is

**Weaviate** is an open-source vector database with a focus on:
- **Hybrid search**: Keyword (BM25) + semantic (vector) in one query
- **Structured data**: Built for knowledge graphs and complex schemas
- **Multiple vectorizers**: Can use OpenAI, Cohere, or Gemini embeddings

Think of it as: *"PostgreSQL meets vector search"*

### Key Feature: Hybrid Search (Preview)

Weaviate can search using both keyword and vector similarity simultaneously. This is critical because:

```
Vector-only: "car" ≠ "automobile" (different words)
Keyword-only: "apple" hits both fruit and company

Hybrid: Get best of both worlds!
```

### Installation

**Option 1: Docker (Recommended for learning)**

```bash
docker run -d -p 8080:8080 -p 50051:50051 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e DEFAULT_VECTORIZER_MODULE=text2vec-transformers \
  semitechnologies/weaviate:latest
```

**Option 2: Weaviate Cloud (Free tier)**

Sign up at [weaviate.io](https://weaviate.io), create a cluster, get your API key.

**Python dependencies:**

```bash
pip install weaviate-client langchain-weaviate
```

### Basic Example: LangChain + Weaviate

```python
from langchain.vectorstores import Weaviate
from langchain_google_generativeai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import weaviate

# Connect to Weaviate (local Docker)
client = weaviate.Client("http://localhost:8080")

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Prepare documents with metadata
docs = [
    Document(page_content="Python is a programming language used for AI development",
             metadata={"category": "programming"}),
    Document(page_content="Machine learning models learn patterns from data",
             metadata={"category": "ai"}),
    Document(page_content="Neural networks are inspired by biological neurons",
             metadata={"category": "ai"}),
    Document(page_content="JavaScript runs in web browsers",
             metadata={"category": "programming"}),
    Document(page_content="Data science combines statistics and programming",
             metadata={"category": "data"}),
]

# Create Weaviate vector store
vector_store = Weaviate.from_documents(
    documents=docs,
    embedding=embeddings,
    client=client,
    by_tenant="BIAPune"  # Organize by tenant (optional)
)

# Simple vector search
results = vector_store.similarity_search("learning from data", k=2)
print("=== Weaviate Search Results ===")
for result in results:
    print(f"- {result.page_content} (category: {result.metadata['category']})")

# Metadata filtering
results_filtered = vector_store.similarity_search(
    "programming languages",
    k=5,
    where_filter={
        "path": ["category"],
        "operator": "Equal",
        "valueString": "programming"
    }
)
print(f"\n=== Filtered Results (programming only) ===")
for result in results_filtered:
    print(f"- {result.page_content}")
```

### Hybrid Search with Weaviate

Hybrid search combines keyword search (BM25) with semantic search. We'll cover this in detail in Section 7.

```python
# Weaviate automatically supports hybrid search
# when you configure it with text2vec module
# We'll see full example in Section 7
```

### When to Use Weaviate

✅ **Perfect for:**
- Hybrid search requirements (keyword + vector)
- Structured, knowledge-graph-like data
- Open-source preference
- Self-hosted deployments
- Medium scale (10K - 100M vectors)

❌ **Not suitable for:**
- Quick learning (more complex setup)
- Fully managed, zero-ops preference
- Extremely high throughput (1M+ qps)

### Weaviate Pricing

- **Self-hosted**: Free (you manage servers)
- **Weaviate Cloud Services (WCS)**: $200-400/month for medium deployments

---

## 5. Pinecone — The Managed Powerhouse

### What It Is

**Pinecone** is a fully managed serverless vector database. You don't manage servers—Pinecone handles all infrastructure.

Think of it as: *"AWS RDS for vector databases"*

### Key Feature: Infinite Scale

Pinecone scales to billions of vectors with:
- Sub-33ms p99 latency at 10M vectors
- Automatic sharding
- High availability (99.95% SLA)
- Zero infrastructure management

### Installation & Setup

```bash
pip install pinecone-client langchain-pinecone
```

**Get API Key:**
1. Sign up at [pinecone.io](https://pinecone.io)
2. Create a free or paid index
3. Copy your API key

### Basic Example: LangChain + Pinecone

```python
from langchain.vectorstores import Pinecone
from langchain_google_generativeai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import pinecone
import os

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=api_key, environment="gcp-starter")

# Create or get index
index_name = "bia-pune-documents"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=768,  # Gemini embeddings dimension
        metric="cosine"
    )

index = pinecone.Index(index_name)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Prepare documents
docs = [
    Document(page_content="Python is a programming language used for AI development",
             metadata={"category": "programming"}),
    Document(page_content="Machine learning models learn patterns from data",
             metadata={"category": "ai"}),
    Document(page_content="Vector databases store embeddings for fast retrieval",
             metadata={"category": "databases"}),
    Document(page_content="LangChain helps build LLM applications",
             metadata={"category": "frameworks"}),
    Document(page_content="Embeddings convert text into numbers",
             metadata={"category": "ai"}),
]

# Create Pinecone vector store
vector_store = Pinecone.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=index_name
)

# Search
results = vector_store.similarity_search("machine learning and data", k=2)
print("=== Pinecone Search Results ===")
for result in results:
    print(f"- {result.page_content}")

# Search with metadata filtering
results_filtered = vector_store.similarity_search(
    "AI applications",
    k=5,
    filter={"category": "ai"}
)
print(f"\n=== Filtered Results (AI only) ===")
for result in results_filtered:
    print(f"- {result.page_content} (category: {result.metadata['category']})")
```

### When to Use Pinecone

✅ **Perfect for:**
- Production applications at scale
- Don't want to manage infrastructure
- Need instant scalability
- High query throughput (1000+ qps)
- Multi-region deployment

❌ **Not suitable for:**
- Tight budget (starts at ~$500/month for 10M vectors)
- Prefer open-source
- Learning/experimentation (overkill)

### Pinecone Pricing

| Tier | Vectors | Cost | Best For |
|------|---------|------|----------|
| **Starter** | 100K | Free | Learning |
| **Standard** | 10M | ~$500/mo | Production startup |
| **Enterprise** | 1B+ | Custom | Large scale |

---

## 6. Head-to-Head Comparison

### Feature Comparison Matrix

| Feature | Chroma | Weaviate | Pinecone |
|---------|--------|----------|----------|
| **Setup** | pip install | Docker/Cloud | Sign up online |
| **Cost** | Free | $0-400/mo | $0-5000+/mo |
| **Learning curve** | Minutes | Hours | Minutes |
| **Metadata filtering** | ✅ Native | ✅ Native | ✅ Native |
| **Hybrid search** | ❌ No | ✅ Built-in | ❌ No (but possible) |
| **Scalability** | 1M vectors | 100M vectors | 1B+ vectors |
| **Latency (p99)** | 100-500ms | 50-200ms | <33ms |
| **Multi-tenancy** | Basic | Good | Excellent |
| **Ops overhead** | None | Medium | Zero |
| **Open source** | ✅ Yes | ✅ Yes | ❌ No |
| **GraphQL API** | ❌ No | ✅ Yes | ❌ REST only |
| **Built-in vectorizers** | ❌ No | ✅ Yes | ❌ No |

### Performance at Different Scales

```
          Chroma  |  Weaviate  | Pinecone
         ---------|------------|----------
1K vecs:   ✓✓✓   |    ✓✓✓    |   ✓✓✓
10K vecs:  ✓✓✓   |    ✓✓✓    |   ✓✓✓
100K vecs: ✓✓    |    ✓✓✓    |   ✓✓✓
1M vecs:   ✓     |    ✓✓     |   ✓✓✓
10M vecs:  ✗     |    ✓✓     |   ✓✓✓
100M vecs: ✗     |    ✓      |   ✓✓✓
1B vecs:   ✗     |    ✗      |   ✓✓✓
```

### Decision Flowchart

```
                 START
                   |
        Is this learning/POC?
           /                    \
         YES                     NO
          |                       |
        CHROMA              Production scale?
          |                 /              \
          |              YES                NO
          |               |                 |
          |        Budget is tight?    Budget flexible?
          |          /           \        /        \
          |        YES          NO     YES        NO
          |         |            |      |          |
          |         |        PINECONE  PINECONE  WEAVIATE
          |         |            |      |
          └─────────└────────────┘      |
               |                        |
          ┌────┴───────────────────┐    |
          |                        |    |
        Scale   Need Hybrid        |    |
       < 1M?    Search?            |    |
        /\       /\                |    |
      YES NO   YES NO              |    |
       |   |    |   |              |    |
       |   |    |   └──PINECONE────┘    |
       |   |    |                       |
       |   |    └──WEAVIATE────────PINECONE
       |   |
       |   └──WEAVIATE (self-hosted)
       |
       └──CHROMA (persisted)
```

### Pricing Comparison (Realistic Example)

**Scenario:** 10M vectors, 100 queries/second, 6 months

```
Chroma (self-hosted):
  - Infrastructure: $2000/mo (1 server) × 6 = $12,000
  - Engineer time: 4 weeks = $16,000 (setup + monitoring)
  TOTAL: ~$28,000

Weaviate (self-hosted):
  - Infrastructure: $1000-2000/mo × 6 = $6,000-12,000
  - Engineer time: 4 weeks = $16,000
  TOTAL: ~$22,000-28,000

Weaviate (WCS managed):
  - WCS cost: $300/mo × 6 = $1,800
  - Engineer time: 1 week = $4,000
  TOTAL: ~$5,800

Pinecone (managed):
  - Pinecone cost: $500-800/mo × 6 = $3,000-4,800
  - Engineer time: 0 (fully managed)
  TOTAL: ~$3,000-4,800
```

### When to Choose Each

| Choose Chroma If... | Choose Weaviate If... | Choose Pinecone If... |
|-------------------|---------------------|---------------------|
| You're learning | Need hybrid search | Need zero ops |
| POC stage | Like open-source | Budget allows |
| <1M vectors | Self-hosted preference | <33ms latency critical |
| No budget | Want GraphQL API | Scaling to 100M+ vectors |

---

## 7. Hybrid Search Explained

### The Problem with Single-Strategy Search

**Vector Search Only:**
```
Query: "What is an automobile?"
Vector embedding might NOT match "car"
Result: Miss relevant documents about cars
```

**Keyword Search Only:**
```
Query: "intelligence"
Returns: Both "artificial intelligence" AND "intelligence agency"
Result: Get irrelevant documents
```

### The Solution: Hybrid Search

Hybrid search combines two complementary strategies:

1. **BM25 (Keyword/Lexical Search)**: Exact term matching
2. **Vector Search (Semantic)**: Meaning-based matching

Then merges results intelligently.

### How BM25 Works (Brief)

BM25 scores documents by:
- How often query terms appear (term frequency)
- How rare those terms are (inverse document frequency)
- Document length normalization

Example:
```
Document 1: "The quick brown fox"
Document 2: "fox fox fox fox fox"

Query: "fox"

BM25 favors Document 1 (normalized by length)
Raw count would favor Document 2
```

### How Vector Search Works (Recap)

```
Query: "intelligent systems"
     ↓
Embed to 768-dim vector
     ↓
Find cosine similarity with all document vectors
     ↓
Return top-k by similarity score
```

### Merging Results: Reciprocal Rank Fusion (RRF)

RRF is a simple, elegant way to combine two ranked lists:

```
RRF Score = 1/(k + rank_in_list)

Where k is constant (usually 60)
```

**Example:**

```
BM25 Results:          Vector Results:
1. Doc A (score 9.2)   1. Doc C (score 0.85)
2. Doc C (score 8.1)   2. Doc A (score 0.81)
3. Doc B (score 7.5)   3. Doc B (score 0.78)

RRF Combination:
Doc A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325 ← BEST
Doc C: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325 ← BEST
Doc B: 1/(60+3) + 1/(60+3) = 0.0159 + 0.0159 = 0.0318

Final Ranking: Doc A/C (tied), then Doc B
```

### Hybrid Search Example: Weaviate

Weaviate has built-in hybrid search:

```python
from langchain.vectorstores import Weaviate
from langchain_google_generativeai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import weaviate

# Connect to Weaviate
client = weaviate.Client("http://localhost:8080")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Documents for hybrid search
docs = [
    Document(page_content="Python is a programming language for AI"),
    Document(page_content="Java is used for building enterprise applications"),
    Document(page_content="Machine learning requires good programming skills"),
    Document(page_content="Deep learning is a subset of machine learning"),
    Document(page_content="Python machine learning frameworks include scikit-learn and TensorFlow"),
]

# Create vector store
vector_store = Weaviate.from_documents(
    documents=docs,
    embedding=embeddings,
    client=client,
    by_tenant="HybridSearchDemo"
)

# Hybrid search (requires Weaviate to be configured with text2vec)
# Weaviate will combine BM25 and vector search automatically
query = "machine learning with Python"

# LangChain interface (hybrid happens behind scenes in Weaviate)
results = vector_store.similarity_search(query, k=2)

print("=== Hybrid Search Results ===")
for i, result in enumerate(results, 1):
    print(f"{i}. {result.page_content}")
```

### Hybrid Search with Custom Merging (Manual RRF)

If using a vector DB without built-in hybrid search:

```python
from langchain.vectorstores import Chroma
from langchain_google_generativeai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

docs = [
    Document(page_content="Python machine learning development"),
    Document(page_content="Java programming enterprise applications"),
    Document(page_content="Machine learning frameworks scikit-learn"),
]

vector_store = Chroma.from_documents(docs, embeddings)

# Manual hybrid search with RRF
def hybrid_search(query, vector_store, k=2):
    """Combine keyword search with vector search using RRF"""
    
    # Vector search
    vector_results = vector_store.similarity_search(query, k=10)
    vector_scores = {i: 1/(60+rank) for rank, i in enumerate(vector_results)}
    
    # Simple keyword search (basic—production would use BM25)
    query_terms = query.lower().split()
    keyword_scores = {}
    
    for idx, doc in enumerate(vector_store._collection.get()['documents']):
        if any(term in doc.lower() for term in query_terms):
            keyword_scores[doc] = sum(
                doc.lower().count(term) for term in query_terms
            )
    
    # RRF merge (simplified)
    merged = {}
    for doc in vector_scores:
        merged[doc] = vector_scores.get(doc, 0) * 0.5
    
    # Return top results
    sorted_results = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_results[:k]]

results = hybrid_search("machine learning Python", vector_store)
print("=== Hybrid Results ===")
for result in results:
    print(f"- {result.page_content}")
```

### When to Use Hybrid Search

Use hybrid search when:
- ✅ You have both exact keywords and semantic concepts
- ✅ Domain has many synonyms (medical, legal documents)
- ✅ Users expect exact term matching + meaning
- ✅ Your vector DB supports it (Weaviate, Elasticsearch)

Stick with vector-only when:
- ✅ Pure semantic search (e.g., image search)
- ✅ All users think semantically
- ✅ No domain jargon to match exactly

---

## 8. Migration Between Vector DBs

### Why Migrate?

Common reasons:
1. **Outgrew current DB** (Chroma → Weaviate)
2. **Cost optimization** (Pinecone → Weaviate self-hosted)
3. **Feature need** (Chroma → Weaviate for hybrid search)
4. **Scale requirements** (Weaviate → Pinecone for billions)

### The LangChain Advantage

LangChain provides a consistent interface across vector DBs:

```python
# Same code works for all three!
vector_store = [Chroma | Weaviate | Pinecone].from_documents(
    documents=docs,
    embedding=embeddings,
    ...
)

results = vector_store.similarity_search(query, k=5)
```

This makes migration much simpler—change the backend, not your application code.

### Step-by-Step Migration Pattern

**Step 1: Load documents from source DB**

```python
from langchain.vectorstores import Chroma
from langchain_google_generativeai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Load from existing Chroma
source_vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Extract documents
all_docs = []
for doc in source_vector_store._collection.get()['documents']:
    all_docs.append(Document(page_content=doc))

print(f"Loaded {len(all_docs)} documents from source")
```

**Step 2: Export to portable format**

```python
import json

# Export as JSON (simple)
export_data = [
    {
        "content": doc.page_content,
        "metadata": doc.metadata or {}
    }
    for doc in all_docs
]

with open("documents_export.json", "w") as f:
    json.dump(export_data, f, indent=2)

print("Exported to documents_export.json")
```

**Step 3: Import into new DB**

```python
from langchain.vectorstores import Weaviate
from langchain.schema import Document
import json
import weaviate

# Load exported data
with open("documents_export.json", "r") as f:
    export_data = json.load(f)

# Recreate documents
docs = [
    Document(
        page_content=item["content"],
        metadata=item["metadata"]
    )
    for item in export_data
]

# Create in target DB (Weaviate)
client = weaviate.Client("http://localhost:8080")
vector_store = Weaviate.from_documents(
    documents=docs,
    embedding=embeddings,
    client=client,
    by_tenant="MigratedData"
)

print(f"Imported {len(docs)} documents to Weaviate")
```

**Step 4: Validate migration**

```python
# Test a few queries on both systems
test_queries = [
    "machine learning",
    "vector databases",
    "Python AI"
]

print("\n=== Validation: Comparing Results ===")
for query in test_queries:
    source_results = source_vector_store.similarity_search(query, k=3)
    target_results = vector_store.similarity_search(query, k=3)
    
    source_docs = [r.page_content[:30] for r in source_results]
    target_docs = [r.page_content[:30] for r in target_results]
    
    match_rate = len(set(source_docs) & set(target_docs)) / 3 * 100
    
    print(f"Query '{query}': {match_rate:.0f}% match")
```

### Complete Migration Example: FAISS → Chroma

```python
from langchain.vectorstores import FAISS, Chroma
from langchain_google_generativeai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Step 1: Load from FAISS
faiss_store = FAISS.load_local(
    "faiss_index",
    embeddings
)

# Step 2: Extract all documents
print("Loading documents from FAISS...")
documents = []
# Extract documents from FAISS (implementation varies)
# For this example, we'll use some sample docs
documents = [
    Document(page_content="Vector search is fast"),
    Document(page_content="Machine learning is data-driven"),
    Document(page_content="Deep learning uses neural networks"),
]

# Step 3: Create in Chroma
print("Creating Chroma vector store...")
chroma_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_migration_db"
)

# Step 4: Validate
print("Validating migration...")
test_query = "neural networks"
faiss_result = faiss_store.similarity_search(test_query, k=1)[0].page_content
chroma_result = chroma_store.similarity_search(test_query, k=1)[0].page_content

print(f"\nFAISS result: {faiss_result}")
print(f"Chroma result: {chroma_result}")
print("Migration complete!")
```

### Best Practices for Migration

| Practice | Why | How |
|----------|-----|-----|
| **Test with sample** | Catch issues early | Migrate 1000 docs first, validate |
| **Verify recall** | Ensure search quality didn't degrade | Compare top-5 results for 10+ queries |
| **Compare latency** | Understand performance trade-offs | Measure p99 latency on both systems |
| **Run in parallel** | Zero downtime transition | Keep old DB live during migration |
| **Document metadata** | Don't lose important info | Export/import all metadata fields |
| **Gradual rollout** | Safe deployment | Migrate 10% users first, monitor |

---

## 9. 2026 Trend: Vector Search in Traditional Databases

### The Shift

**2023-2024:** Specialized vector databases gain traction  
**2025-2026:** Traditional databases adding vector support

This is important: You might not need a specialized vector DB anymore!

### PostgreSQL + pgvector

PostgreSQL added native vector support with the **pgvector** extension.

**Setup:**
```bash
# Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

# Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    category VARCHAR(50),
    embedding vector(768)  -- 768-dimensional vector
);

# Create index for fast search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
```

**Why this matters:**
- ✅ Keep documents and vectors in one place
- ✅ Use SQL for hybrid queries
- ✅ Leverage PostgreSQL ecosystem (existing monitoring, backups)
- ✅ Cheaper than managed vector DBs

**When to use pgvector:**
- You already use PostgreSQL
- Data is highly relational
- Need ACID transactions
- Can accept slightly higher latency

### MongoDB Atlas Vector Search

MongoDB added vector search directly to document database.

```javascript
// Index documents with vector field
db.documents.createIndex({
    embedding: "cosmosSearch"
});

// Query with vector search
db.documents.findOne({
    $search: {
        cosmosSearch: {
            vector: [query_embedding_array],
            k: 5
        },
        returnStoredSource: true
    }
});
```

**Why this matters:**
- ✅ Documents are already JSON—add vectors without reshaping
- ✅ Use same database for documents + vectors
- ✅ Leverage MongoDB ecosystem

**When to use MongoDB vector search:**
- You already use MongoDB
- Documents are complex/nested JSON
- Don't need dedicated vector performance
- Want simplicity over max performance

### Specialized vs Traditional: Decision Matrix

| Factor | PostgreSQL+pgvector | MongoDB+VectorSearch | Specialized (Chroma/Pinecone) |
|--------|-------------------|---------------------|------------------------------|
| **Setup complexity** | Medium | Medium | Low |
| **Performance** | Good | Good | Excellent |
| **Latency (p99)** | 50-200ms | 50-200ms | 10-50ms |
| **Max vectors** | 100M | 100M | 1B+ |
| **Cost** | $500/mo server | $1000/mo | $500+/mo |
| **Existing DB users** | Yes → pgvector | Yes → MongoDB Vector | No preference |
| **Learning ease** | Medium (SQL+pgvector) | Medium (MongoDB+aggregation) | Easy |

### Quick Comparison

```
Use Traditional DB + Vector If:
✓ You already use PostgreSQL/MongoDB
✓ Data is highly relational/nested
✓ Want single database for everything
✓ Latency can be 100-200ms
✓ <100M vectors

Use Specialized Vector DB If:
✓ Starting fresh (no existing DB)
✓ Need ultra-low latency (<50ms)
✓ Scaling to 1B+ vectors
✓ Hybrid search is essential (Weaviate)
✓ Fully managed infrastructure (Pinecone)
```

---

## 10. Hands-On Exercises

### Exercise 1: Build a Document Search System with Chroma

**Objective:** Create a simple document search system using Chroma.

**Setup:**
```bash
pip install chromadb langchain langchain-google-generativeai
```

**Task:**
1. Create 5 documents about different topics
2. Add them to Chroma
3. Query by similarity
4. Filter by metadata

**Starter Code:**

```python
import chromadb
from langchain_google_generativeai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma

# Initialize
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Create documents (5 different topics)
documents = [
    Document(
        page_content="Python is a versatile programming language for AI and data science",
        metadata={"category": "programming", "difficulty": "beginner"}
    ),
    Document(
        page_content="Machine learning algorithms learn patterns from training data",
        metadata={"category": "ai", "difficulty": "intermediate"}
    ),
    Document(
        page_content="Neural networks are inspired by biological brain structures",
        metadata={"category": "ai", "difficulty": "advanced"}
    ),
    Document(
        page_content="Docker containerization helps deploy applications reliably",
        metadata={"category": "devops", "difficulty": "intermediate"}
    ),
    Document(
        page_content="Vector databases enable fast semantic search at scale",
        metadata={"category": "databases", "difficulty": "advanced"}
    ),
]

# TODO: Create Chroma vector store from documents
# Hint: Use Chroma.from_documents()

# TODO: Search 1 - Find documents similar to "learning algorithms"
# Print results with category

# TODO: Search 2 - Find documents in "ai" category similar to "deep networks"
# Hint: Use similarity_search_with_score() to see relevance scores

# TODO: Bonus - Add metadata filtering
# Find all "intermediate" difficulty documents about programming or AI

print("✓ Exercise 1 complete!")
```

**Solution:**
```python
import chromadb
from langchain_google_generativeai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

documents = [
    Document(page_content="Python is a versatile programming language for AI and data science",
             metadata={"category": "programming", "difficulty": "beginner"}),
    Document(page_content="Machine learning algorithms learn patterns from training data",
             metadata={"category": "ai", "difficulty": "intermediate"}),
    Document(page_content="Neural networks are inspired by biological brain structures",
             metadata={"category": "ai", "difficulty": "advanced"}),
    Document(page_content="Docker containerization helps deploy applications reliably",
             metadata={"category": "devops", "difficulty": "intermediate"}),
    Document(page_content="Vector databases enable fast semantic search at scale",
             metadata={"category": "databases", "difficulty": "advanced"}),
]

# Create Chroma vector store
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_exercise1"
)

# Search 1: General similarity search
results = vector_store.similarity_search("learning algorithms", k=2)
print("=== Search 1: 'learning algorithms' ===")
for result in results:
    print(f"- {result.page_content}")
    print(f"  Category: {result.metadata['category']}\n")

# Search 2: With relevance scores
results_with_scores = vector_store.similarity_search_with_relevance_scores(
    "deep networks", k=3
)
print("=== Search 2: 'deep networks' (with scores) ===")
for result, score in results_with_scores:
    print(f"- {result.page_content[:50]}... (score: {score:.2f})")
    print(f"  Category: {result.metadata['category']}\n")

# Bonus: Filter by difficulty and search
print("=== Bonus: Intermediate difficulty in 'ai' or 'programming' ===")
for doc in documents:
    if (doc.metadata['difficulty'] == 'intermediate' and 
        doc.metadata['category'] in ['ai', 'programming']):
        print(f"- {doc.page_content}")

print("\n✓ Exercise 1 complete!")
```

### Exercise 2: Compare Search Results Across Vector DBs

**Objective:** Compare FAISS vs Chroma to see differences.

**Task:**
1. Create same dataset in both FAISS and Chroma
2. Run same query on both
3. Compare results (order, scores)
4. Discuss why they differ

**Starter Code:**

```python
from langchain.vectorstores import FAISS, Chroma
from langchain_google_generativeai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Same documents for both
documents = [
    Document(page_content="Machine learning is a subset of artificial intelligence"),
    Document(page_content="Deep learning uses neural networks with multiple layers"),
    Document(page_content="AI helps automate repetitive tasks"),
    Document(page_content="Neural networks learn from data iteratively"),
    Document(page_content="Artificial intelligence is transforming technology"),
]

# TODO: Create FAISS vector store
# faiss_store = FAISS.from_documents(...)

# TODO: Create Chroma vector store
# chroma_store = Chroma.from_documents(...)

# TODO: Run same query on both
query = "learning with neural networks"

# TODO: Compare results
# Print results from FAISS and Chroma side by side

# TODO: Analyze:
# - Do they return same documents?
# - Is the order the same?
# - What explains any differences?

print("✓ Exercise 2 complete!")
```

**Analysis Questions:**
1. Are the top results identical? Why/why not?
2. Does order differ? What causes this?
3. Which seems more semantically correct to you?
4. When would you prefer FAISS over Chroma?

---

## 11. Quick Reference Card

### Installation Cheat Sheet

```bash
# Chroma
pip install chromadb

# Weaviate (Python client)
pip install weaviate-client
docker run -d -p 8080:8080 semitechnologies/weaviate:latest

# Pinecone
pip install pinecone-client

# LangChain integrations
pip install langchain-chroma langchain-weaviate langchain-pinecone
pip install langchain-google-generativeai
```

### Code Template: LangChain Integration

```python
from langchain_google_generativeai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

# Initialize embeddings (same for all DBs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Create documents
docs = [
    Document(page_content="...", metadata={...}),
    Document(page_content="...", metadata={...}),
]

# ----- CHROMA -----
from langchain.vectorstores import Chroma
chroma = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
results = chroma.similarity_search("query", k=5)

# ----- WEAVIATE -----
from langchain.vectorstores import Weaviate
import weaviate
client = weaviate.Client("http://localhost:8080")
weaviate_db = Weaviate.from_documents(docs, embeddings, client=client)
results = weaviate_db.similarity_search("query", k=5)

# ----- PINECONE -----
from langchain.vectorstores import Pinecone
import pinecone
pinecone.init(api_key="...", environment="gcp-starter")
pinecone_db = Pinecone.from_documents(docs, embeddings, index_name="index")
results = pinecone_db.similarity_search("query", k=5)
```

### Embedding Dimensions

| Model | Dimension | Size | Speed |
|-------|-----------|------|-------|
| Gemini | 768 | Small | Fast |
| OpenAI ada-002 | 1536 | Large | Medium |
| Cohere | 1024 | Medium | Medium |

### When to Choose: Quick Decision Tree

```
START
  └─ Is this for learning?
      ├─ YES → CHROMA (free, instant)
      └─ NO → Need self-hosted?
          ├─ YES (cost sensitive) → WEAVIATE
          └─ NO (budget available) → PINECONE
```

### Common Operations

```python
# Vector store creation
vector_store = DB.from_documents(documents, embeddings, ...)

# Search
results = vector_store.similarity_search(query, k=5)

# Search with scores
results = vector_store.similarity_search_with_relevance_scores(query, k=5)

# Metadata filtering (varies by DB)
results = vector_store.similarity_search(
    query, 
    k=5,
    filter={"category": "ai"}  # Weaviate/Pinecone syntax
)

# Add documents post-creation
vector_store.add_documents([new_docs])

# Delete documents
vector_store.delete([doc_ids])

# Persistence
vector_store.persist()  # Chroma
# (Weaviate/Pinecone persistent by default)
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| **Chroma slow with 1M vectors** | Switch to Weaviate/Pinecone |
| **Weaviate high latency** | Check Docker/network, add index |
| **Pinecone high cost** | Start with Starter tier, migrate later |
| **Metadata filtering not working** | Check filter syntax for your DB |
| **Embeddings dimension mismatch** | Ensure all embedding models are same dimension |
| **Migration memory error** | Process documents in batches (1000 at a time) |

---

## Summary & Key Takeaways

### What You've Learned

1. **Vector DBs >= FAISS** for production: Persistence, filtering, APIs, scaling
2. **Three categories**: Embedded (Chroma), Self-hosted (Weaviate), Managed (Pinecone)
3. **Chroma**: Perfect for learning, small POCs
4. **Weaviate**: Best for hybrid search, structured data, cost efficiency
5. **Pinecone**: Production powerhouse, scales to billions, zero ops
6. **Hybrid Search**: Combine keyword + semantic for best results
7. **Migration**: Use LangChain abstractions for easy switching
8. **2026 Trend**: Traditional DBs + vectors (pgvector, MongoDB)

### Decision Framework

| Scenario | Choose |
|----------|--------|
| Learning vector search | Chroma |
| POC in 2-3 weeks | Chroma |
| Need hybrid search | Weaviate |
| Self-hosted, cost-sensitive | Weaviate |
| Production, billions of vectors | Pinecone |
| Already have PostgreSQL | pgvector |
| Existing MongoDB user | MongoDB Vector Search |

### What's Next

**Session 12:** Building RAG systems with vector databases  
**Session 13:** Production deployment patterns

---

## Appendix: Lab Setup Guide

### Chroma Local Setup
```bash
pip install chromadb
python -c "import chromadb; print('Chroma ready')"
```

### Weaviate Docker Setup
```bash
docker pull semitechnologies/weaviate:latest
docker run -d -p 8080:8080 -p 50051:50051 \
  semitechnologies/weaviate:latest

# Verify
curl http://localhost:8080/v1/.well-known/ready
```

### Pinecone Cloud Setup
```bash
# Sign up at pinecone.io
# Get API key from console
# Create index: Name "demo", Dimension 768, Metric cosine
export PINECONE_API_KEY="your-key-here"
python -c "import pinecone; pinecone.whoami()"
```

### Verify All Three Work

```python
import os
os.environ['GOOGLE_API_KEY'] = 'your-google-api-key'

from langchain_google_generativeai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

test_embedding = embeddings.embed_query("test")
print(f"✓ Embeddings working ({len(test_embedding)} dimensions)")

# Test each DB
print("✓ Chroma: pip install chromadb")
print("✓ Weaviate: docker run semitechnologies/weaviate:latest")
print("✓ Pinecone: pinecone.io API ready")
```

---

**Last Updated:** April 2026  
**Audience:** BEGINNER students, BIA Pune  
**Duration:** 3 hours  
**Prerequisites:** Session 10 (Vector Search 101), basic Python
