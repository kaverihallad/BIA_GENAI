# Build Guide: Smart Study Assistant
## From Skeleton to Working App — Using AI to Write Real Code

---

## What You're Building

A command-line Smart Study Assistant that can:
- Answer questions about your study notes using RAG (Retrieval-Augmented Generation)
- Summarize topics on demand
- Generate flashcards for revision
- Quiz you on any topic  
- Self-critique and refine its own answers
- Route queries intelligently to the right handler

**Concepts used from the course:**

| File | Session | Concept |
|------|---------|---------|
| `loader.py` | Session 12 | Document chunking (RecursiveCharacterTextSplitter) |
| `vectorstore.py` | Sessions 10-11 | Embeddings + ChromaDB vector store |
| `retriever.py` | Session 12 | RAG chain with LCEL |
| `tools.py` | Session 9 | @tool decorator, custom tools |
| `evaluator.py` | Sessions 8+12 | Self-reflection + Precision@k, Recall@k, F1 |
| `router.py` | Sessions 9+12 | Query classification and routing |
| `agent.py` | Session 9 | create_react_agent from LangGraph |
| `main.py` | All | CLI bringing everything together |

---

## Prerequisites

- **Google API key** (for Gemini models)
  - Get one at: https://aistudio.google.com/app/apikeys
  - Free tier available with limited usage
- **Python 3.10 or higher**
- **Google Antigravity IDE** (or any AI coding assistant like Claude, ChatGPT, Copilot)
- About **1-2 hours** to complete the full project

---

## Quick Setup (~10 minutes)

### 1. Set Your API Key

Open `config.py` and replace the placeholder:

```python
os.environ.setdefault("GOOGLE_API_KEY", "your-actual-api-key-here")
```

Save the file.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Sample Data

Create a file at `data/sample_notes.txt` with your study material. For example:

```
Machine Learning Basics

Supervised Learning
Supervised learning is a machine learning approach where we train models on 
labeled data. Each training example has an input (features) and a corresponding 
output (label). The model learns the relationship between inputs and outputs.

Examples: Linear regression, logistic regression, decision trees, neural networks

Unsupervised Learning
Unsupervised learning finds patterns in unlabeled data. No target labels exist.
Used for clustering, dimensionality reduction, and anomaly detection.

Examples: K-means clustering, Principal Component Analysis (PCA)
```

---

## Phase 1: The Knowledge Base (TODOs 1-6, ~30 min)

### loader.py — Loading & Chunking Documents

Your study assistant needs to load documents and break them into manageable chunks that fit in the LLM's context window.

#### TODO 1: Load a text file

**What to do:** Implement the `load_text_file()` function to read a file from disk.

**Open Google Antigravity IDE** and navigate to `loader.py`. Ask it:

```
Complete the load_text_file function. It should read a file from file_path 
and return its contents as a string. Use utf-8 encoding. Handle the case 
where the file doesn't exist by raising an error.
```

**What to look for:**
- Does it use `open()` with `encoding="utf-8"`?
- Does it return the file contents as a string?
- Does it properly close the file (or use a context manager)?

**Expected result:**
```python
def load_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
```

#### TODO 2-3: Chunk the text

**What to do:** Implement the `chunk_text()` function to split text into chunks.

In Antigravity, ask:

```
Complete the chunk_text function. Create a RecursiveCharacterTextSplitter with 
chunk_size=CHUNK_SIZE and chunk_overlap=CHUNK_OVERLAP (from config). Then call 
splitter.split_text(text) and return the list of chunks.
```

**What to look for:**
- Does it import `RecursiveCharacterTextSplitter` correctly?
- Does it use `CHUNK_SIZE` and `CHUNK_OVERLAP` from config?
- Does it return a list of strings (the chunks)?

**Why this matters:** Chunking breaks long documents into overlapping pieces. A document about neural networks might become:
- Chunk 1: "Neural networks are inspired by biological neurons. A neuron takes weighted inputs..."
- Chunk 2: "weighted inputs and produces an output. Multiple neurons form a layer..."
- Chunk 3: "A layer contains multiple neurons. Multiple layers form a deep neural network..."

The overlap (CHUNK_OVERLAP=50) ensures concepts aren't split across chunk boundaries.

### vectorstore.py — Embeddings & ChromaDB

The vector store converts text chunks into embeddings (mathematical representations) and stores them for fast retrieval.

#### TODO 4: Initialize embeddings

**What to do:** Create the `get_embeddings()` function to initialize Google's embedding model.

Ask Antigravity:

```
Complete the get_embeddings function. It should return a 
GoogleGenerativeAIEmbeddings instance using EMBEDDING_MODEL from config.
```

**What to look for:**
- Does it use `GoogleGenerativeAIEmbeddings()`?
- Does it pass `model=EMBEDDING_MODEL`?

**Expected result:**
```python
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
```

#### TODO 5: Create vector store

**What to do:** Implement `create_vectorstore()` to store chunk embeddings in ChromaDB.

Ask Antigravity:

```
Complete the create_vectorstore function. Use Chroma.from_texts() to create 
a new vector store. Pass it the chunks list, the embeddings from get_embeddings(), 
and persist_directory=CHROMA_PERSIST_DIR. Return the vectorstore.
```

**What to look for:**
- Does it call `Chroma.from_texts()`?
- Does it pass all three arguments: chunks, embeddings, persist_directory?
- Does it return the vectorstore object?

**Why this matters:** ChromaDB persists your embeddings to disk. Next time you run the app, you don't need to re-embed all the chunks—you can just load the saved store.

#### TODO 6: Load existing vector store

**What to do:** Implement `load_vectorstore()` to load a saved ChromaDB.

Ask Antigravity:

```
Complete the load_vectorstore function. Create a Chroma instance using:
- persist_directory=CHROMA_PERSIST_DIR
- embedding_function=get_embeddings()
Return the loaded vectorstore.
```

### Test Phase 1

Once you've completed TODOs 1-6, test in your terminal:

```bash
python -c "
from loader import load_and_chunk
chunks = load_and_chunk('data/sample_notes.txt')
print(f'Success! Got {len(chunks)} chunks')
"
```

Expected output: `Success! Got 5 chunks` (or however many chunks your file produces)

---

## Phase 2: The RAG Chain (TODOs 7-9, ~20 min)

RAG (Retrieval-Augmented Generation) means: retrieve relevant chunks from your notes, then feed them to the LLM as context so it can answer questions accurately.

### retriever.py — Retrieval + Generation

#### TODO 7: Create retriever

**What to do:** Turn the vectorstore into a retriever that fetches the top-K most relevant chunks.

Ask Antigravity:

```
Complete the retriever creation in build_rag_chain. Create a retriever from 
the vectorstore using as_retriever(search_kwargs={"k": TOP_K}). TOP_K is 
imported from config and specifies how many chunks to retrieve per question.
```

**What to look for:**
- Does it call `vectorstore.as_retriever()`?
- Does it pass `search_kwargs={"k": TOP_K}`?

#### TODO 8: Create RAG prompt template

**What to do:** Build a prompt that tells the LLM to answer based on the retrieved context.

Ask Antigravity:

```
Complete the rag_prompt creation. Use ChatPromptTemplate.from_messages() to 
create a prompt that includes:
1. A system message telling the LLM: "Answer the question based ONLY on the 
   provided context. If the context doesn't contain the answer, say 'I don't 
   have information about that.'"
2. A human message with {context} and {question} placeholders
Make sure to format the context using the format_docs function.
```

**What to look for:**
- Does it use `ChatPromptTemplate.from_messages()`?
- Does it include {context} and {question} as placeholders?
- Does it instruct the LLM to use only the context?

**Example prompt structure:**
```
System: "Answer based ONLY on context..."
Human: "Context: {context}\n\nQuestion: {question}"
```

#### TODO 9: Build the LCEL chain

**What to do:** Connect retriever → prompt → LLM → parser into a single chain.

Ask Antigravity:

```
Build the complete LCEL chain as a single expression. Start with a dict that 
pipes the vectorstore retriever through format_docs into "context", and passes 
the question through RunnablePassthrough() as "question". Then pipe that dict 
into the rag_prompt, through the llm, and finally through StrOutputParser().

The structure should look like:
chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | rag_prompt | llm | StrOutputParser()
```

**Why this matters:** This chain does everything:
1. Takes your question
2. Retrieves relevant chunks (similarity search)
3. Formats them into context
4. Feeds question + context to the prompt
5. Sends it to the LLM
6. Parses the output into a string

### Test Phase 2

```bash
python -c "
from loader import load_and_chunk
from vectorstore import create_vectorstore
from retriever import build_rag_chain

chunks = load_and_chunk('data/sample_notes.txt')
vs = create_vectorstore(chunks)
chain = build_rag_chain(vs)
answer = chain.invoke('What is supervised learning?')
print('Answer:', answer)
"
```

Expected output: A paragraph about supervised learning, based on your notes.

---

## Phase 3: Tools & Agent (TODOs 10-12, 19-21, ~25 min)

Tools are functions the agent can call. The agent decides which tool to use based on the user's request.

### tools.py — Custom Study Tools

#### TODO 10: Summarize tool

**What to do:** Create a tool that generates concise summaries.

Ask Antigravity:

```
Implement the summarize_topic tool. It should:
1. Get the LLM using get_llm()
2. Create a prompt asking: "Provide a concise 3-4 bullet point summary of this 
   topic: {topic}. Focus on key concepts."
3. Call llm.invoke() with the prompt
4. Extract and return the content

The function is already decorated with @tool, you just need to implement the body.
```

**What to look for:**
- Does it use `get_llm().invoke()`?
- Does it ask for 3-4 bullet points?
- Does it return a string?

**Expected usage:**
```
Tool input: "Neural Networks"
Tool output: "• Inspired by biological neurons\n• Made of layers\n• Use backpropagation..."
```

#### TODO 11: Generate flashcards tool

**What to do:** Create a tool that generates study flashcards.

Ask Antigravity:

```
Implement the generate_flashcards tool. It should:
1. Get the LLM using get_llm()
2. Create a prompt asking: "Create 5 study flashcards (Q&A pairs) from this 
   content: {content}. Format each as 'Q: [question]\nA: [answer]'"
3. Call llm.invoke() with the prompt
4. Extract and return the content

The function is already decorated with @tool.
```

**Expected output format:**
```
Q: What are neural networks?
A: Mathematical models inspired by biological neurons...

Q: What is backpropagation?
A: An algorithm for training neural networks...
```

#### TODO 12: Quiz tool

**What to do:** Create a tool that generates quizzes.

Ask Antigravity:

```
Implement the quiz_me tool. It should:
1. Get the LLM using get_llm()
2. Create a prompt asking: "Create a 3-question multiple choice quiz on this 
   topic: {topic}. For each question, provide 4 options (A, B, C, D) and 
   the correct answer."
3. Call llm.invoke() with the prompt
4. Extract and return the content

The function is already decorated with @tool.
```

### agent.py — The Study Agent

The agent uses LangGraph's ReAct (Reasoning + Acting) pattern. It reasons about which tool to use, acts by calling it, and repeats until the task is done.

#### TODOs 19-21: Create the agent

**What to do:** Build the brain of the assistant.

Ask Antigravity:

```
Complete the create_study_agent function:

TODO 19: Initialize the LLM using ChatGoogleGenerativeAI with model=LLM_MODEL 
and temperature=LLM_TEMPERATURE from config.

TODO 20: Get all tools by calling get_all_tools()

TODO 21: Create the agent by calling create_react_agent(model=llm, tools=tools, 
prompt=AGENT_PROMPT). The AGENT_PROMPT is already defined at the top of the file.

The function should return the agent.
```

**What to look for:**
- Does TODO 19 create a ChatGoogleGenerativeAI instance?
- Does TODO 20 call `get_all_tools()`?
- Does TODO 21 call `create_react_agent()` with the right arguments?

**Why ReAct is powerful:** The agent can reason:
> "The user asked 'Quiz me on neural networks'. I should use the quiz_me tool with topic='neural networks'."
> Then it calls the tool, gets the quiz, and returns it to the user.

### Test Phase 3

```bash
python -c "
from agent import create_study_agent, chat_with_agent
agent = create_study_agent()
response = chat_with_agent(agent, 'Summarize neural networks for me')
print(response)
"
```

Expected output: A summary of neural networks from the LLM.

---

## Phase 4: Self-Reflection (TODOs 13-16, ~20 min)

Self-reflection means the LLM critiques its own answers and improves them. This is a quality assurance loop.

### evaluator.py — Critique & Refine

#### TODO 13: Critique response

**What to do:** Implement a function where the LLM evaluates its own answer.

Ask Antigravity:

```
Implement critique_response. It should:
1. Get the LLM using get_llm()
2. Create a prompt that asks: "Critique this answer for accuracy, completeness, 
   and clarity. Question: {question}\n\nAnswer: {answer}\n\nProvide specific 
   feedback on what's missing or unclear."
3. Call llm.invoke() with the prompt
4. Extract and return the content as a string
```

**Expected usage:**
```
Question: "What is supervised learning?"
Answer: "It's a type of learning."
Critique: "The answer is incomplete. It should explain labeled data, training, 
and examples of supervised learning tasks."
```

#### TODO 14: Refine response

**What to do:** Improve the answer based on the critique.

Ask Antigravity:

```
Implement refine_response. It should:
1. Get the LLM using get_llm()
2. Create a prompt that asks: "Improve this answer based on the critique. 
   Question: {question}\n\nOriginal answer: {answer}\n\nCritique: {critique}
   \n\nProvide an improved answer that addresses the critique."
3. Call llm.invoke() with the prompt
4. Extract and return the improved answer
```

**Why this matters:** The code already has a loop (`self_refine`) that calls critique and refine repeatedly. Your implementation makes the loop work.

#### TODO 15-16: Evaluation metrics

These measure how well your RAG system retrieves relevant documents.

**TODO 15: Precision@K**

Ask Antigravity:

```
Implement precision_at_k. This measures: "Of the top-K documents I retrieved, 
how many were actually relevant?"

The formula is: count of relevant in retrieved[:k] divided by k

Hint: retrieved_k = retrieved[:k]
      count = sum(1 for doc in retrieved_k if doc in relevant)
      return count / k
```

**Example:**
- Retrieved: [doc1, doc2, doc3, doc4, doc5]
- Relevant: [doc1, doc3, doc6]
- Precision@3 = 2/3 ≈ 0.67 (docs 1 and 3 are relevant)

**TODO 16: Recall@K**

Ask Antigravity:

```
Implement recall_at_k. This measures: "Of all the relevant documents, 
how many did I find in the top-K?"

The formula is: count of relevant in retrieved[:k] divided by total relevant

Hint: retrieved_k = retrieved[:k]
      count = sum(1 for doc in retrieved_k if doc in relevant)
      total = len(relevant)
      return count / total if total > 0 else 0
```

**Example:**
- Retrieved: [doc1, doc2, doc3, doc4, doc5]
- Relevant: [doc1, doc3, doc6]
- Recall@3 = 2/3 ≈ 0.67 (found 2 of 3 relevant docs)

---

## Phase 5: The Router (TODOs 17-18, ~15 min)

The router intelligently directs queries to the right handler. Instead of always using RAG, it detects what the user wants and routes accordingly.

### router.py — Smart Query Routing

#### TODO 17: Classify query

**What to do:** Ask the LLM to categorize the user's intent.

Ask Antigravity:

```
Implement classify_query. It should:
1. Get the LLM using get_llm()
2. Create a prompt that asks the LLM to classify the question into ONE of these 
   categories: "study_question", "summarize", "flashcards", "quiz", or "general"
3. Explain each category:
   - study_question: Needs information from study notes (use RAG)
   - summarize: User wants a topic summary
   - flashcards: User wants flashcards generated
   - quiz: User wants to be quizzed
   - general: General question that doesn't need notes (use LLM knowledge)
4. Call llm.invoke() with the prompt
5. Extract the response and return ONLY the category name in lowercase
```

**Expected usage:**
```
Input: "What is supervised learning?"
Output: "study_question"

Input: "Summarize neural networks"
Output: "summarize"

Input: "What is 2+2?"
Output: "general"
```

#### TODO 18: Route query

**What to do:** Handle each category appropriately.

Ask Antigravity:

```
Implement route_query. Based on the category from classify_query(), call:
- "study_question" -> rag_chain.invoke(question)
- "summarize" -> tools["summarize"].invoke(question)
- "flashcards" -> tools["flashcards"].invoke(question)
- "quiz" -> tools["quiz"].invoke(question)
- "general" -> get_llm().invoke(question).content

Make sure to handle each case and return the result as a string.
```

---

## Phase 6: Run the Full App! (~10 min)

Now you need a `main.py` file to tie everything together. Create it:

```bash
touch main.py
```

Ask Antigravity to implement a simple CLI:

```
Create main.py with a command-line interface for the Smart Study Assistant.

It should:
1. Load the study notes using load_and_chunk from loader.py
2. Create or load the vectorstore using vectorstore functions
3. Build the RAG chain using build_rag_chain from retriever.py
4. Create the agent using create_study_agent from agent.py
5. Accept user commands:
   - "ask <question>" -> Use RAG to answer
   - "agent <task>" -> Use the agent with tools
   - "quit" -> Exit the app
   - Any other input -> Pass to the agent
6. Print responses clearly

Print a welcome message at startup explaining what the assistant can do.
```

Once you have `main.py`, run it:

```bash
python main.py
```

Try these commands:

```
ask What is supervised learning?
agent Make me flashcards about neural networks
Summarize decision trees for me
Quiz me on machine learning basics
What is 2+2?
quit
```

**Expected behavior:**

| Input | Behavior |
|-------|----------|
| `ask What is supervised learning?` | Uses RAG to answer from your notes |
| `agent Make me flashcards about neural networks` | Agent calls the flashcards tool |
| `Summarize decision trees` | Router detects "summarize", uses summarize tool |
| `Quiz me on ML` | Router detects "quiz", generates a quiz |
| `What is 2+2?` | Router classifies as "general", uses LLM knowledge |

---

## Bonus Challenges

Once the basic version works, try these extensions:

### 1. Add Your Own Study Notes

Replace `data/sample_notes.txt` with real study material. Try it with:
- Math notes
- History facts
- Programming concepts
- Scientific terms

### 2. Add Web Search

Create a new tool that searches the web:

```python
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Use a free API like SerpAPI or DuckDuckGo
    pass
```

Then add it to `get_all_tools()`.

### 3. Add Conversation Memory

Modify `agent.py` to remember previous questions:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
agent = create_react_agent(..., checkpointer=memory)
```

### 4. Build a Streamlit UI

Create a web interface instead of CLI:

```bash
pip install streamlit
```

Create `streamlit_app.py`:

```python
import streamlit as st
from agent import create_study_agent, chat_with_agent

st.title("Smart Study Assistant")
user_input = st.text_input("Ask me anything:")
if user_input:
    agent = create_study_agent()
    response = chat_with_agent(agent, user_input)
    st.write(response)
```

Run with:
```bash
streamlit run streamlit_app.py
```

### 5. Multi-Document Support

Extend the loader to handle multiple files:

```python
def load_all_from_folder(folder_path: str) -> list[str]:
    """Load and chunk all .txt files in a folder."""
    import os
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            chunks = load_and_chunk(os.path.join(folder_path, filename))
            all_chunks.extend(chunks)
    return all_chunks
```

---

## Concept Map: What You Built

This project uses every major concept from the course:

| What You Did | Course Session | Real-World Application |
|-------------|---------------|----------------------|
| Loaded & chunked documents | Session 12 | Any document processing pipeline (OCR, web scraping, data ingestion) |
| Created embeddings | Session 10 | Semantic search, recommendation systems, similarity matching |
| Stored in ChromaDB | Session 11 | Knowledge bases, enterprise search, vector databases |
| Built RAG chain with LCEL | Session 12 | Chatbots, Q&A systems, customer support bots, documentation assistants |
| Created @tool functions | Session 9 | Extending AI agents with capabilities (calculator, email, database access) |
| Built a ReAct agent | Session 9 | Autonomous AI assistants, multi-step reasoning, task planning |
| Self-reflection loop | Session 8 | Quality assurance, fact-checking, content moderation |
| Evaluation metrics | Session 12 | Monitoring production AI systems, benchmarking, A/B testing |
| Query routing | Sessions 9+12 | Multi-skill assistants, cost optimization, specialized handlers |

You've built a **real Python application**—not a Jupyter notebook—using professional software patterns that power production AI systems.

---

## Troubleshooting

### Issue: "No module named 'langchain_google_genai'"
**Solution:** Run `pip install -r requirements.txt` again, or `pip install langchain-google-genai`

### Issue: "Google API key not set"
**Solution:** Make sure you've set the API key in `config.py`:
```python
os.environ.setdefault("GOOGLE_API_KEY", "YOUR_ACTUAL_KEY")
```

### Issue: "No chunks found" or "Vectorstore is empty"
**Solution:** 
1. Verify `data/sample_notes.txt` exists and has content
2. Try deleting `chroma_db` folder to reset the database
3. Run Phase 1 tests again to verify chunking works

### Issue: Agent returns "I don't have the tools" 
**Solution:** Make sure `get_all_tools()` in `tools.py` returns a non-empty list:
```python
def get_all_tools():
    return [summarize_topic, generate_flashcards, quiz_me]
```

### Issue: LLM responses are empty or truncated
**Solution:** Try increasing `LLM_TEMPERATURE` in `config.py` from 0 to 0.7 for more creative responses.

---

## Next Steps

Once you've completed the Smart Study Assistant:

1. **Deploy it:** Turn it into a Streamlit web app or FastAPI backend
2. **Scale it:** Add support for PDFs, images, videos
3. **Integrate it:** Connect to your school's LMS (Canvas, Blackboard)
4. **Personalize it:** Add user profiles and learning goals
5. **Evaluate it:** Measure accuracy of answers against ground truth
6. **Share it:** Deploy to Hugging Face Spaces or Replit

You now have a foundation for building AI-powered learning tools. The architecture you've learned—document loading, embeddings, retrieval, routing, and agents—applies to almost any domain: customer support, legal research, medical diagnosis assistance, code generation, and more.

**Keep building!**
