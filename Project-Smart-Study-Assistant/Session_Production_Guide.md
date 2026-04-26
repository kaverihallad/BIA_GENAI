# Session: Architecture, Production Thinking & Interview Prep
## Smart Study Assistant Deep-Dive

**Institution:** Boston Institute of Analytics, Pune  
**Level:** Beginner (Post-Project Completion)  
**Duration:** 3 hours  
**Audience:** Students who have completed the Smart Study Assistant project (CLI + Streamlit UI)

---

## Overview

This session answers the critical question: **"Why is the code structured this way?"**

Students just spent 2-3 sessions building a 10-file project with LLMs, ChromaDB, and LangChain. They followed a template, filled in TODOs, and it "just worked." But they may not understand:

- Why split into 10 files instead of one 500-line script?
- What breaks if we change the architecture?
- How does this scale to real production systems?
- What would interviewers ask about this design?
- How do the same patterns apply across totally different domains?

By the end of this session, they should be able to:
1. **Explain** why each file exists and what it does
2. **Trace** dependencies through the import chain
3. **Predict** what breaks if architecture changes
4. **Recognize** these patterns in job interviews and real systems
5. **Apply** the same architecture to different use cases
6. **Discuss** production concerns: env vars, error handling, caching, logging

---

## Part 1: Architecture Deep-Dive (45 minutes)

### Objective
Students understand that good architecture means:
- Each file has ONE clear responsibility
- Changes in one area don't ripple everywhere
- You can swap the "face" (UI) without touching the "brain" (logic)

### 1.1 The 10 Files: What Each One Does

#### Core Configuration
| File | Purpose | Owns |
|------|---------|------|
| **config.py** | Single source of truth | API keys, model names, hyperparameters (chunk size, top-k, temperature), file paths |

**Key Principle:** If you need to change the model from Gemini to Claude, or chunk size from 500 to 1000, you change it ONCE in config.py. Everything else auto-reads it.

#### Data Pipeline
| File | Purpose | Owns |
|------|---------|------|
| **loader.py** | Read raw documents | `load_text_file()` — opens .txt files, handles encoding |
| | | `chunk_text()` — uses RecursiveCharacterTextSplitter |
| | | `load_and_chunk()` — orchestrates both |
| **vectorstore.py** | Embeddings + persistence | `get_embeddings()` — initializes Google Generative AI embedder |
| | | `create_vectorstore()` — Chroma.from_texts() → persists to disk |
| | | `load_vectorstore()` — reads existing Chroma DB |
| **retriever.py** | RAG chain construction | `get_llm()` — initializes ChatGoogleGenerativeAI |
| | | `format_docs()` — joins retrieved chunks into prompt |
| | | `build_rag_chain()` — LCEL: retriever → prompt → LLM → parser |
| | | `ask_question()` — invoke the chain |

#### Logic Layer
| File | Purpose | Owns |
|------|---------|------|
| **tools.py** | Tool definitions | `@tool summarize_topic()` — asks LLM for summaries |
| | | `@tool generate_flashcards()` — creates Q&A pairs |
| | | `@tool quiz_me()` — generates 3-question quiz |
| | | `get_all_tools()` — registry for agent to discover |
| **router.py** | Query classification | `classify_query()` — "Is this a study_question, summarize, flashcards, quiz, or general?" |
| | | `route_query()` — dispatches to right handler (RAG vs tool vs LLM) |
| **evaluator.py** | Quality control | `critique_response()` — self-reflection: "Is this good?" |
| | | `refine_response()` — improve based on critique |
| | | `self_refine()` — loop: generate → critique → refine |
| | | `precision_at_k()`, `recall_at_k()` — evaluation metrics |

#### Agent & Orchestration
| File | Purpose | Owns |
|------|---------|------|
| **agent.py** | LangGraph agent | `create_study_agent()` — ReAct agent with tools |
| | | `chat_with_agent()` — invoke with a message |

#### Entry Points (Same Backend, Different UIs)
| File | Purpose | Owns |
|------|---------|------|
| **main.py** | CLI interface | Banner, help menu, command loop |
| | | Loads vectorstore, builds RAG chain |
| | | Handles: "ask", "agent", "summarize", "flashcards", "quiz", "eval", "index" |
| **app.py** | Streamlit UI | Page config, sidebar, session state |
| | | Same modules as main.py, different "skin" |

---

### 1.2 The Import Chain: How Dependencies Flow

Draw this on whiteboard or show a diagram:

```
config.py
  ↓
loader.py ← (imports CHUNK_SIZE, CHUNK_OVERLAP)
vectorstore.py ← (imports EMBEDDING_MODEL, CHROMA_PERSIST_DIR, COLLECTION_NAME)
retriever.py ← (imports LLM_MODEL, LLM_TEMPERATURE, TOP_K)
  ↓
tools.py ← (imports get_llm from retriever)
router.py ← (imports get_llm from retriever)
evaluator.py ← (imports get_llm from retriever)
  ↓
agent.py ← (imports get_all_tools from tools)
  ↓
main.py ← (imports everything)
app.py ← (imports everything)
```

**Key Insight:** The dependency graph is **acyclic**. No circular imports. This is intentional design, not accident.

- config.py has NO imports from the project (it's a "leaf")
- loader, vectorstore, retriever build on top of config
- tools, router, evaluator depend on retriever
- agent depends on tools
- main and app depend on everything

**Why This Matters:**
- To test vectorstore.py, you only need config.py and ChromaDB
- To test retriever.py, you need config, vectorstore, but NOT main or app
- If app.py breaks, you can still run main.py — they're independent
- You can swap app.py for a Discord bot UI without touching retriever.py

---

### 1.3 Two Entry Points, One Backend

**main.py:**
```python
# Load and index
chunks = load_and_chunk(data_file)
vectorstore = create_vectorstore(chunks)

# Build RAG
rag_chain = build_rag_chain(vectorstore)

# Interactive loop
while True:
    user_input = input("You > ")
    if "ask" in user_input:
        answer = ask_question(rag_chain, question)
```

**app.py:**
```python
# Streamlit init
st.set_page_config(...)

# Load and index (cached with @st.cache_resource)
vectorstore = load_vectorstore()

# Build RAG (cached)
rag_chain = build_rag_chain(vectorstore)

# Interactive widgets
user_input = st.text_input("Ask a question")
if user_input:
    answer = ask_question(rag_chain, user_input)
    st.write(answer)
```

**The Core Lesson:** Both call the SAME functions from retriever.py. The UI layer is 100% decoupled from the logic layer.

This is why:
- You can test retriever.py without Streamlit
- You can deploy main.py on a server and app.py on the web, both using the same vectorstore
- If you need a third UI (Discord bot, API), you don't rewrite retriever — you write a new entry point

---

### 1.4 Why @tool Matters

In **tools.py**:
```python
@tool
def summarize_topic(topic: str) -> str:
    """Create a concise summary of a study topic."""
    # ... implementation
```

The `@tool` decorator:
- Is NOT just a label
- Tells LangGraph: "This is a callable capability"
- Creates metadata (name, docstring, args) that the agent can read
- Lets the agent DISCOVER what it can do without hardcoding

**Without @tool:** You'd have to manually list tools in agent code.  
**With @tool:** Agent reads function signatures and docstrings.

```python
# What the agent sees:
{
    "name": "summarize_topic",
    "description": "Create a concise summary of a study topic.",
    "args": {"topic": "string"}
}
```

---

### 1.5 LCEL as Plumbing

In **retriever.py**:
```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

The pipe operator `|` chains components like Unix pipes:

```
retriever      → format docs into string
question       → pass through unchanged
    ↓
rag_prompt     → insert into template
    ↓
llm            → send to model
    ↓
StrOutputParser → extract text from response
```

**Why This Matters:**
- Easy to visualize the flow
- You can insert steps: `retriever | format_docs | rerank_docs | format_docs_again`
- Composable: reuse chains
- Same syntax as Unix: `cat file | grep pattern | wc -l`

---

### 1.6 "What Would Break If..." Exercise (15 min)

**Scenario 1:** "What if we hardcoded the API key in retriever.py instead of config.py?"

```python
# BAD: in retriever.py
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key="AIzaSyB9v4K_..."  # hardcoded!
)
```

**What breaks:**
- Can't share code on GitHub (secret exposed)
- Can't run tests with a test key
- Can't run locally vs. production without editing code
- Team member with different key can't run it
- Every file that imports retriever has the secret

**Correct:** Store in config.py and environment.

---

**Scenario 2:** "What if retriever.py also handled chunking (from loader.py)?"

**Before (Good):**
```
loader.py → chunks text
retriever.py → builds RAG chain
main.py → calls both in order
```

**After (Bad):**
```
retriever.py → chunks AND builds RAG
main.py → just calls retriever
```

**What breaks:**
- Can't index with different chunk sizes without editing retriever
- Can't test chunking independently
- If you want to use the same chunks with a different vectorstore (FAISS instead of Chroma), retriever is tightly coupled
- Two concerns (chunking + RAG) mixed in one file makes it harder to find bugs

**Correct:** Keep chunking separate. Each file has ONE job.

---

**Scenario 3:** "What if we put everything in one 500-line main.py?"

```python
# main.py (bad)
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
# ... 20 more imports ...

# All the embedding code
def get_embeddings():
    ...

# All the chunking code
def chunk_text(text):
    ...

# All the RAG code
def build_rag_chain(vectorstore):
    ...

# All the tool code
@tool
def summarize_topic(topic):
    ...

# 500 lines later...
if __name__ == "__main__":
    ...
```

**What breaks:**
- Impossible to find anything (where is summarize_topic?)
- Impossible to test retrieval without loading tools
- Impossible to reuse retriever in a different project
- One typo or bad import and the whole thing fails
- Team members step on each other's code
- Hard to debug: is the error in chunking or RAG?

**Correct:** Split into modules. 10 small files beat 1 big file.

---

### 1.7 Summary: The Architecture Principles

| Principle | How the Project Uses It |
|-----------|------------------------|
| **Separation of Concerns** | Each file has one job (load, embed, retrieve, tool, route, evaluate, orchestrate, UI) |
| **Dependency Injection** | config.py is injected everywhere, not hardcoded |
| **Acyclic Dependencies** | You can trace imports in one direction (config → everything, tools → agent → main) |
| **Decoupled UIs** | main.py and app.py are NOT interdependent; both call the same backend |
| **Composable Chains** | LCEL chains are reusable (rag_chain, agent, evaluator) |
| **Single Responsibility** | router.py doesn't also vectorize, retriever.py doesn't also evaluate |
| **Declarative Tools** | @tool decorator lets agent auto-discover capabilities |

---

## Part 2: Production Patterns (30 minutes)

### Objective
Students understand that "production" doesn't mean "complex DevOps." It means:
- Your code doesn't break when real data arrives
- Secrets stay secret
- Errors don't cascade
- The system is observable (you know what's happening)
- It's maintainable (6 months from now, you remember why you did it)

### 2.1 Environment Variables: Never Hardcode Secrets

**Bad (in config.py):**
```python
os.environ.setdefault("GOOGLE_API_KEY", "AIzaSyB9v4K_Y21c52hkrw3jK7eDup3yM0xwKHk")
```

**Problem:** Commit this to GitHub and your key is public. Anyone can use it, and Google will bill you.

**Better (.env file):**
```bash
# .env (never committed to GitHub)
GOOGLE_API_KEY=AIzaSyB9v4K_Y21c52hkrw3jK7eDup3yM0xwKHk
LLM_MODEL=gemini-2.5-flash
CHROMA_PERSIST_DIR=./chroma_db
```

**config.py:**
```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env")

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")  # default fallback
```

**What to Put in .env:**
- API keys (Google, OpenAI, Hugging Face)
- Database passwords (never hardcode!)
- Sensitive paths
- Feature flags (DEBUG=True/False)

**What to Put in config.py:**
- Non-sensitive defaults (CHUNK_SIZE, TOP_K, TEMPERATURE)
- Model names (these are public)
- Feature toggles (ENABLE_CACHING=True)

**.gitignore (always):**
```
.env
.env.local
__pycache__/
*.pyc
chroma_db/
venv/
.DS_Store
```

**Why This Matters:**
- Developers have different keys (local vs. staging vs. production)
- You can deploy to production without changing code
- Your code is safe to share on GitHub
- New team members can run `cp .env.example .env` and fill in their own keys

---

### 2.2 Error Handling: Graceful Degradation

**Bad (in main.py):**
```python
answer = ask_question(rag_chain, question)  # If this fails, whole app crashes
```

**Better:**
```python
try:
    answer = ask_question(rag_chain, question)
except ValueError as e:
    answer = f"I couldn't answer that. Error: {e}"
except Exception as e:
    answer = "Something went wrong. Please try again."
    print(f"[ERROR] Unexpected: {e}")  # Log it
```

**Pattern:**
```python
try:
    # Attempt the main operation
    result = risky_operation()
except SpecificException as e:
    # Handle the specific case
    result = fallback_value
    logger.error(f"Specific error: {e}")
except Exception as e:
    # Catch-all for unknown errors
    result = generic_fallback
    logger.exception(f"Unexpected error: {e}")
finally:
    # Clean up (close files, release locks)
    cleanup()
```

**In retriever.py:**
```python
def ask_question(chain, question: str) -> str:
    """Ask a question using the RAG chain."""
    try:
        return chain.invoke(question)
    except TimeoutError:
        return "The AI model timed out. Please try a shorter question."
    except Exception as e:
        print(f"[ERROR] {e}")
        return "I'm having trouble answering right now. Try again in a moment."
```

**In tools.py:**
```python
@tool
def summarize_topic(topic: str) -> str:
    """Create a concise summary of a study topic."""
    try:
        llm = get_llm()
        response = llm.invoke(f"Summarize: {topic}")
        return response.content
    except Exception as e:
        return f"Couldn't summarize. Please check your topic and try again."
```

**Why This Matters:**
- User gets a friendly message instead of a stack trace
- You know WHAT failed (logs show the error)
- System keeps running (one bad query doesn't crash the whole app)
- Production apps EXPECT failures (network down, API rate-limited, memory full)

---

### 2.3 Caching: Expensive Operations Only Once

**In Streamlit (app.py):**
```python
import streamlit as st
from retriever import build_rag_chain
from vectorstore import load_vectorstore

# Load vectorstore ONCE, not on every button click
@st.cache_resource
def get_vectorstore():
    return load_vectorstore()

# Build RAG chain ONCE
@st.cache_resource
def get_rag_chain():
    vectorstore = get_vectorstore()
    return build_rag_chain(vectorstore)

# In your app
vectorstore = get_vectorstore()
rag_chain = get_rag_chain()
```

**When to Cache:**
- Model loading (heavy)
- Vectorstore creation (disk I/O)
- API client initialization
- Expensive computations

**When NOT to Cache:**
- User input (changes every time)
- Timestamps
- File paths that change
- Data that gets updated frequently

**Invalidation:**
```python
# Clear cache if data changed
if st.button("Re-index Notes"):
    get_vectorstore.clear()
    get_rag_chain.clear()
    st.success("Cache cleared!")
```

**In CLI (main.py):**
```python
# Python doesn't have built-in caching like Streamlit
# But you can cache manually:

_vectorstore_cache = None

def get_vectorstore():
    global _vectorstore_cache
    if _vectorstore_cache is None:
        _vectorstore_cache = load_vectorstore()
    return _vectorstore_cache
```

**Why This Matters:**
- Model loading takes 2-5 seconds; you don't want that on every interaction
- Vectorstore is 50MB+; you load it once from disk
- Without caching, Streamlit reloads everything on every button press
- Production systems use Redis, Memcached for large-scale caching

---

### 2.4 Logging vs. Printing

**Bad (in retriever.py):**
```python
def build_rag_chain(vectorstore):
    print("Building RAG chain...")  # Debugging noise
    # ... code ...
    print("Chain built successfully!")
    return chain
```

**Better:**
```python
import logging

logger = logging.getLogger(__name__)

def build_rag_chain(vectorstore):
    logger.info("Building RAG chain...")
    try:
        # ... code ...
        logger.info("Chain built successfully!")
        return chain
    except Exception as e:
        logger.error(f"Failed to build chain: {e}")
        raise
```

**Logging Levels:**
- `DEBUG` — Detailed info for developers (variable values, function calls)
- `INFO` — General flow ("Starting vectorstore load")
- `WARNING` — Something unexpected but not fatal ("Chunk size very large")
- `ERROR` — Something failed ("API key invalid")
- `CRITICAL` — System is unusable ("Database down")

**Setup (in main.py):**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Write to file
        logging.StreamHandler(),  # Also print to console
    ]
)

logger = logging.getLogger(__name__)
logger.info("Smart Study Assistant started")
```

**In Production:**
```python
# Use different levels for dev vs. production
import os

if os.getenv("DEBUG") == "True":
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)  # Less verbose in prod
```

**Why This Matters:**
- Logs can be written to files, sent to monitoring systems (e.g., Datadog, CloudWatch)
- You can change log level without editing code
- `print()` output gets lost in Streamlit reruns
- Production systems parse logs for errors and alerts
- You can search logs for debugging ("find all errors from 3 PM to 4 PM")

---

### 2.5 Project Structure & .gitignore

**Recommended layout:**
```
smart-study-assistant/
├── config.py              # Configuration (no secrets)
├── loader.py
├── vectorstore.py
├── retriever.py
├── tools.py
├── router.py
├── evaluator.py
├── agent.py
├── main.py               # CLI entry point
├── app.py                # Streamlit entry point
├── requirements.txt      # Dependencies
├── .env.example          # Example env file (no secrets!)
├── .gitignore            # What NOT to commit
├── .github/
│   └── workflows/        # CI/CD (if needed)
├── data/
│   └── sample_notes.txt  # Example data
├── tests/
│   ├── test_loader.py
│   ├── test_retriever.py
│   └── test_tools.py
├── logs/                 # (if using file logging)
└── README.md             # How to run the project
```

**.gitignore:**
```
# Environment
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
venv/
.venv/

# Project-specific
chroma_db/              # Vectorstore (regenerated from data/)
*.log                   # Log files
*.db                    # Databases
.DS_Store               # macOS
.idea/                  # IDE files

# Secrets (NEVER commit)
secrets.json
credentials.yaml
```

**Why This Matters:**
- Never accidentally commit secrets
- Keeps repo clean (no __pycache__, no .pyc files)
- Vectorstore is regenerated from data, no need to commit
- New developers don't get old log files
- CI/CD systems have clean checkouts

---

### 2.6 The "Works on My Machine" Problem: requirements.txt

**Bad (no requirements.txt):**
```
Dev says: "Just install langchain, chromadb, streamlit"
But which versions? The one from 2024? 2025?
Student installs: langchain==0.2.1 (2024) vs. langchain==0.3.0 (2025)
Different behavior, incompatibilities, frustration.
```

**Good (requirements.txt with versions):**
```
langchain-core==0.1.52
langchain-google-genai==1.0.5
langchain-community==0.1.39
langchain-text-splitters==0.0.1
langgraph==0.0.64
chromadb==0.4.24
colorama==0.4.6
streamlit==1.31.1
```

**How to Create:**
```bash
pip freeze > requirements.txt
```

**How to Use:**
```bash
pip install -r requirements.txt
```

**Why Specific Versions Matter:**
- langchain 0.3 changed APIs from 0.2
- chromadb 0.5 changed schema from 0.4
- streamlit 1.32 might deprecate a feature you use
- Specific versions = reproducible across machines and time

**For Development (with ranges):**
```
langchain-core>=0.1.50,<0.2
langchain-google-genai>=1.0.0,<1.1
```

**Why This Matters:**
- Your project works the same way on your machine, the student's machine, the production server
- You can onboard a new team member: "git clone, python -m venv venv, pip install -r requirements.txt, python main.py"
- No surprises from version mismatches

---

### 2.7 Summary: Production Checklist

| Item | What to Do | Why |
|------|-----------|-----|
| Secrets | Store in .env, not config.py | Never commit API keys |
| Errors | try/except at boundaries, user-friendly messages | Graceful degradation |
| Caching | @st.cache_resource for expensive ops | Performance |
| Logging | Use logging module, not print() | Searchable, leveled, goes to files |
| Structure | Separate concerns, acyclic deps | Maintainability |
| .gitignore | Exclude .env, __pycache__, chroma_db/ | Clean repos |
| requirements.txt | Freeze versions | Reproducibility |

---

## Part 3: Use Case Walkthroughs (45 minutes)

### Objective
Students see that the architecture is a **template**. The same 10-file structure applies to:
- Customer support bots
- Legal document analyzers
- Company knowledge bases

**Key Insight:** Only the data, prompts, and tools change. The structure stays the same.

### Use Case 1: Customer Support Bot

**Scenario:** A SaaS company wants an AI bot to answer customer support tickets using their product docs, FAQs, and past ticket solutions.

**Data Source:**
- Product documentation (PDFs)
- FAQ page (HTML)
- Ticket database (CSV: question, solution)

**Architecture Mapping:**

| File | Smart Study | Support Bot | How It Changes |
|------|-------------|------------|-----------------|
| config.py | GOOGLE_API_KEY, CHUNK_SIZE=500, TOP_K=3 | GOOGLE_API_KEY, CHUNK_SIZE=1000 (longer docs), TOP_K=5 (more context) | Hyperparameters tuned for support docs |
| loader.py | `load_text_file()` for .txt | `load_pdf_file()`, `load_csv_file()`, `load_html_file()` | Multiple doc types |
| vectorstore.py | Chroma collection: "study_notes" | Chroma collection: "support_docs" | Different collection names, same infrastructure |
| retriever.py | RAG prompt: "Based on study notes, answer..." | RAG prompt: "You are a support agent. Based on our docs, resolve this ticket..." | Prompt is customized |
| tools.py | `summarize_topic()`, `generate_flashcards()`, `quiz_me()` | `escalate_ticket()`, `check_order_status()`, `create_refund_request()` | Domain-specific tools |
| router.py | Routes: study_question, summarize, flashcards, quiz, general | Routes: billing_issue, technical_support, general_inquiry, escalation | Domain-specific categories |
| evaluator.py | Critique answers for accuracy, completeness | Critique support responses for tone (empathy), accuracy, completeness | Same pattern, different criteria |
| agent.py | ReAct agent with study tools | ReAct agent with support tools | Same agent framework, different tools |
| main.py | CLI: "ask", "agent", "quiz" | CLI: "ticket", "agent", "escalate" | Different commands |
| app.py | Streamlit: text input, session history | Streamlit: ticket viewer, response widget, escalation button | Different UI, same backend |

**Example: The RAG Prompt Change**

Study Assistant:
```python
rag_prompt = ChatPromptTemplate.from_template("""
You are a Smart Study Assistant. Answer the question based ONLY on the provided study notes.
If the notes don't contain the answer, say "I don't have that information."

Context:
{context}

Question:
{question}

Answer:
""")
```

Support Bot:
```python
rag_prompt = ChatPromptTemplate.from_template("""
You are a professional support agent for [Company]. Your goal is to resolve customer issues quickly and empathetically.

Use the product documentation and past solutions below to answer. If a solution requires escalation, say so.

Context:
{context}

Customer Question:
{question}

Solution:
""")
```

**Same chain, different personality.**

---

### Use Case 2: Legal Document Analyzer

**Scenario:** A law firm wants to search contracts, flag risks, and compare clauses across documents.

**Data Source:**
- Contract library (PDFs)
- Case law database (PDFs)
- Compliance checklists (DOCX)

**Key Difference:** Data privacy and local processing.

| File | Study Assistant | Legal Analyzer | Why Different |
|------|-----------------|----------------|----------------|
| config.py | Uses Google's embedding API | Uses local embeddings (sentence-transformers) | Contracts are confidential; can't send to Google's servers |
| loader.py | Loads .txt files | Loads PDFs with OCR (pypdf) | Legal docs are mostly PDFs |
| vectorstore.py | Chroma (fine), Google embeddings | Chroma (fine), local embeddings | No cloud API calls for secrets |
| retriever.py | Same RAG structure | Same RAG structure, but filtered by doc type | Can add: "search only in contracts, not case law" |
| tools.py | study_tools | `find_clause()`, `compare_contracts()`, `flag_risk()`, `extract_terms()` | Domain-specific legal tools |
| router.py | Routes to summarize, quiz, etc. | Routes to contract_review, clause_search, compliance_check | Legal-specific workflows |
| evaluator.py | Evaluates correctness, completeness | Evaluates correctness, ALSO ensures no hallucinations (legal errors = liability!) | Higher standards |
| agent.py | Same agent framework | Same agent framework | No change |
| main.py | "ask", "quiz", "summarize" | "search_contracts", "flag_risks", "compare" | Different commands |
| app.py | Streamlit text input | Streamlit: document upload, risk dashboard, contract comparison view | Richer UI |

**Critical Change: Evaluator**

```python
def evaluate_legal_response(question: str, answer: str) -> dict:
    """For legal docs, hallucinations are dangerous."""
    llm = get_llm()
    
    critique = llm.invoke(f"""
    Review this legal analysis for:
    1. Accuracy (no made-up case names or clause numbers)
    2. Completeness (all relevant clauses mentioned?)
    3. Risk (did we miss a critical issue?)
    
    If anything is uncertain, flag it with [UNCERTAIN].
    
    Analysis: {answer}
    """)
    
    if "[UNCERTAIN]" in critique:
        return {"approved": False, "reason": "Contains uncertain elements"}
    return {"approved": True}
```

Study Assistant is forgiving (missing an answer is okay). Legal is strict (wrong answer = lawsuit).

---

### Use Case 3: Internal Knowledge Base

**Scenario:** A 500-person company wants employees to search meeting notes, Confluence docs, Slack threads, and team wikis.

**Data Source:**
- Confluence pages (API)
- Slack threads (API)
- Google Drive docs (API)
- Internal wikis (web scrape)

**Architecture:**

| File | Study Assistant | Knowledge Base | Why Different |
|------|-----------------|----------------|----------------|
| config.py | Reads local files | Plus: Confluence API key, Slack token, refresh schedule | Added external API credentials |
| loader.py | `load_text_file()` → reads disk | `load_confluence()`, `load_slack()`, `load_gdrive()` → API clients | Multiple data sources |
| vectorstore.py | Persist to `./chroma_db` | Persist to shared location (cloud storage or network drive) | Multiple users, shared index |
| retriever.py | Same RAG | Add metadata filtering: "search only in Sales channel" | Access control |
| tools.py | summarize_topic, quiz_me | `find_expert()`, `search_by_team()`, `get_recent_updates()` | Domain-specific tools |
| router.py | Routes to quiz, summarize | Routes to document_search, team_finder, recent_news | Different flows |
| evaluator.py | Evaluates correctness | Evaluates correctness, ALSO tracks usage analytics | Who searched what? |
| agent.py | Same | Same | No change |
| main.py | CLI | CLI (internal tool) | Same |
| app.py | Streamlit | Streamlit with user auth (who am I?) | Authentication added |

**Production Extras for Knowledge Base:**

1. **Scheduled Re-indexing:**
```python
# In a cron job or Lambda
import schedule
import time

def reindex_docs():
    print("Fetching latest Confluence pages...")
    docs = load_confluence()
    vectorstore = create_vectorstore(docs)
    print("Index updated at", datetime.now())

schedule.every().day.at("02:00").do(reindex_docs)  # 2 AM
while True:
    schedule.run_pending()
    time.sleep(60)
```

2. **Access Control:**
```python
# In retriever.py
def build_rag_chain_with_acl(vectorstore, user_id: str):
    """Only search docs the user has access to."""
    # Filter vectorstore by user groups
    restricted_vectorstore = filter_by_acl(vectorstore, user_id)
    # Build chain on restricted store
    return build_rag_chain(restricted_vectorstore)
```

3. **Usage Analytics:**
```python
# In evaluator.py or logger
def log_search(user_id: str, query: str, result_count: int):
    """Track what people search for."""
    analytics.log({
        "user": user_id,
        "query": query,
        "timestamp": datetime.now(),
        "results_returned": result_count,
    })
    # Later: "Most searched topics = training needs"
```

---

### 3.4 Summary Table: The Architecture is Reusable

| Layer | Component | Changes Per Use Case |
|-------|-----------|----------------------|
| **Data** | loader.py, vectorstore.py | Different file types, APIs, access controls |
| **Retrieval** | retriever.py | Different prompts, different metadata filtering |
| **Logic** | tools.py, router.py | Domain-specific tools and categories |
| **Quality** | evaluator.py | Different evaluation criteria (legal is stricter than study) |
| **Orchestration** | agent.py | Rarely changes (LangGraph is general-purpose) |
| **UI** | main.py, app.py | Different commands, different widgets, different auth |

**What Never Changes:**
- The 10-file structure
- The import chain (config at bottom, app at top)
- The dependency hierarchy

**What Always Changes:**
- Data sources and loaders
- Prompts (study vs. support vs. legal voice)
- Tools (what the agent can do)
- Routes (how to classify queries)

---

## Part 4: Interview Scenarios (45 minutes)

### Objective
Students can explain their project to interviewers and handle follow-up questions.

**This section is an overview. Full Q&A is in the separate Interview Scenarios document.**

### 4.1 Sample Interview Questions

**Softball:**
1. "Walk me through your project. What does it do?"
2. "How are the 10 files organized?"
3. "Why is config.py separate from the other files?"

**Architecture:**
4. "How would you add a new tool (e.g., generate_study_notes)?"
5. "What would break if you moved vectorstore creation into retriever.py?"
6. "How do main.py and app.py share code without duplicating it?"

**Production:**
7. "How do you handle API keys in your project?"
8. "What happens if the LLM API is down?"
9. "How would you scale this to 1000 concurrent users?"

**Trade-offs:**
10. "Why ChromaDB and not FAISS?"
11. "Why use an agent instead of just RAG?"
12. "What are the limits of your current architecture?"

**Full answers in the Interview Scenarios document.**

---

### 4.2 How to Practice

**Interview Prep Steps:**
1. **Explain once** — Tell the story naturally (5 min)
2. **Explain again** — Diagram the architecture on paper (5 min)
3. **Predict questions** — "What might they ask about...?" (5 min)
4. **Answer out loud** — Speak to yourself or a peer (10 min per question)
5. **Refine** — Record yourself, listen, improve (10 min)

**Record yourself with:**
- Voice memo on phone
- OBS (free, records screen + audio)
- Loom (free, browser-based)

**Listen for:**
- Long pauses (you don't have an answer; okay, move on)
- Filler words ("um", "like", "basically"; cut them)
- Jargon without explanation ("I used LCEL" → "I used LangChain's pipe syntax")
- Logical jumps (make the story easy to follow)

---

## Instructor Pacing Guide (3 hours)

| Part | Duration | Deliverable |
|------|----------|-------------|
| **Part 1: Architecture Deep-Dive** | 45 min | Students can explain why 10 files |
| 1.1 The 10 Files (table) | 10 min | Reference table printed |
| 1.2 The Import Chain (diagram) | 5 min | Draw on board, students copy |
| 1.3 Two Entry Points (code walkthrough) | 8 min | Show main.py and app.py, highlight same imports |
| 1.4 Why @tool Matters | 5 min | Run a tool, show agent's discovery |
| 1.5 LCEL as Plumbing | 5 min | Draw the pipe chain |
| 1.6 "What Would Break If..." Exercise | 12 min | Scenario 1, 2, 3 (3-4 min each) |
| **Part 2: Production Patterns** | 30 min | Students understand "production" = practices |
| 2.1 Environment Variables (live demo) | 6 min | Show .env, config.py loading, hardcoded bad |
| 2.2 Error Handling (code review) | 5 min | Show try/except patterns |
| 2.3 Caching (demo) | 5 min | Load model twice, cache vs no cache |
| 2.4 Logging | 4 min | Show log level output |
| 2.5 Project Structure | 4 min | Show tree layout, .gitignore |
| 2.6 requirements.txt | 2 min | Show version pinning |
| **Part 3: Use Case Walkthroughs** | 45 min | Students see architecture reuse |
| 3.1 Customer Support Bot (walkthrough + table) | 15 min | Show how only tools/prompts change |
| 3.2 Legal Analyzer (walkthrough + table) | 15 min | Show local embeddings, stricter eval |
| 3.3 Knowledge Base (walkthrough + table) | 15 min | Show scheduling, access control, analytics |
| **Q&A / Buffer** | 20 min | Clarify, re-explain, deeper dives |
| **Interview Prep** | 15 min | Reference the Interview Scenarios doc, do 1-2 mock questions |
| **Total** | 180 min | — |

---

## Materials Needed

### Printed/Projected
- [ ] Table 1.1: The 10 Files
- [ ] Diagram 1.2: Import Chain (whiteboard or slide)
- [ ] Code snippets for 1.3 (main.py vs app.py)
- [ ] Exercise questions for 1.6
- [ ] Production Checklist (Table 2.7)
- [ ] Use Case Mapping Tables (3.1, 3.2, 3.3)

### Live Demos
- [ ] Terminal: Show .env vs hardcoded API key
- [ ] Python: Load model with/without caching (time difference)
- [ ] Logging: Change log level, show output
- [ ] Agent: @tool decorator, agent auto-discovers tools

### Handouts
- [ ] Architecture Deep-Dive (1-page summary)
- [ ] Production Patterns Checklist
- [ ] Use Case Template (blank for students to fill in for their own project idea)
- [ ] Interview Scenarios Link

---

## Key Takeaways (Summary Slide)

1. **Architecture = Sustainability**
   - 10 files beats 1 file
   - Separation of concerns = flexibility
   - Acyclic dependencies = debuggable

2. **Production = Professionalism**
   - Never hardcode secrets
   - Graceful error handling
   - Logging, not printing
   - Pin your versions

3. **Patterns Repeat**
   - Same 10-file structure works for support bots, legal docs, knowledge bases
   - Only the data, tools, and prompts change
   - This is how real companies build AI systems

4. **Interview Ready**
   - You can explain "why" (not just "how")
   - You've seen the patterns in production
   - You can think about trade-offs
   - You know what breaks and why

---

## Appendix: Common Questions

**Q: Why not use FastAPI instead of Streamlit?**  
A: FastAPI is for APIs (backend); Streamlit is for web UIs (frontend). You can use both: FastAPI backend + Streamlit frontend. The architecture stays the same.

**Q: What if we need real-time collaboration (multiple users at once)?**  
A: The architecture scales:
- Vectorstore → move to a shared service (Pinecone, Weaviate, cloud Chroma)
- Sessions → move to a session store (Redis, database)
- Models → use a model-serving platform (Modal, Replicate, cloud inference)
- UI stays the same (Streamlit, React, whatever)

**Q: Can we use a different LLM (Claude, LLaMA)?**  
A: Change 3 lines:
- config.py: New model name
- retriever.py: `from langchain_anthropic import ChatAnthropic` instead of ChatGoogleGenerativeAI
- Tools: Might need to adjust prompts (Claude prefers different phrasing)
The rest is unchanged.

**Q: Why didn't we use Pydantic BaseModel for schemas?**  
A: For beginners, it's unnecessary complexity. Once you're comfortable, add it for:
- Validation (ensure user input is the right type)
- Documentation (schema = API docs)
- Type hints (IDE autocomplete)

**Q: How do we deploy this to production?**  
A: Many paths:
- CLI: Package with PyInstaller, ship as executable
- Streamlit: `streamlit cloud` (free tier), Heroku, EC2
- API: Wrap retriever.py in FastAPI, deploy to cloud (AWS, GCP, Azure)
The architecture doesn't change; you just change how you run it.

---

## Resources for Instructors

- Interview Scenarios document (full Q&A)
- Architecture Visual (HTML interactive diagram)
- BUILD_GUIDE.md (how to construct the project)
- STREAMLIT_GUIDE.md (how to build the Streamlit UI)
- Sample .env.example file (template for students)

---

**End of Session Guide**

---

## Quick Reference: File-by-File Checklist

Use this during the session to keep students on track:

```
✓ config.py         → All settings in one place (API key, chunk size, model)
✓ loader.py         → Read files, chunk them
✓ vectorstore.py    → Create/load embeddings + Chroma
✓ retriever.py      → Build RAG chain, ask questions
✓ tools.py          → @tool functions for the agent
✓ router.py         → Classify queries, dispatch to handler
✓ evaluator.py      → Critique and refine answers
✓ agent.py          → Create ReAct agent with tools
✓ main.py           → CLI entry point (uses all above)
✓ app.py            → Streamlit entry point (uses all above)

Dependency direction: config → loaders → retriever → tools/router → agent → main/app
No circles. Clean.
```

---

**Session ends. Students should feel:**
- Confidence in explaining their project
- Understanding of "why" the architecture is designed this way
- Awareness of production concerns
- Ability to adapt the pattern to new domains
