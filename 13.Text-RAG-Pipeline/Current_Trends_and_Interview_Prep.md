# GenAI & Agentic AI: Current Trends + Interview Prep
### Quick Reference for BIA Students — April 2026

---

## Part 1: What's Hot Right Now (April 2026)

### 1. Agentic AI Is the Main Event

Single-purpose chatbots are out. **Multi-agent orchestration** — teams of specialized AI agents coordinating on tasks — is the dominant paradigm. Gartner reported a **1,445% surge** in multi-agent system inquiries from Q1 2024 to Q2 2025, and adoption has only accelerated since.

What this means for you: the skills you're learning (tool use, manager/worker patterns, LangGraph) are exactly what companies are hiring for.

### 2. Agentic RAG > Naive RAG

Plain "retrieve then generate" is now considered naive RAG. The industry has moved to **Agentic RAG** — where agents decide *what* to retrieve, *when* to retrieve, and *whether* the answer is good enough. Key features: adaptive retrieval (skip retrieval for simple questions), self-reflection loops, multi-source routing, and real-time data feeds.

**Adaptive RAG** is now mandatory for cost control — it reserves heavy compute for complex queries only.

### 3. MCP (Model Context Protocol) Is Everywhere

Anthropic's MCP has become a standard for connecting LLMs to external tools and data. Think of it as the "USB-C for AI" — a universal way for agents to plug into APIs, databases, file systems, and services. Knowing how to build and consume MCP servers is a growing skill demand.

### 4. Multi-Agent Orchestration Frameworks

The landscape in April 2026:
- **LangGraph** — the production standard for stateful multi-agent workflows
- **CrewAI** — popular for quick multi-agent setups with role-based agents
- **AutoGen (Microsoft)** — strong in enterprise/research settings
- **OpenAI Swarm** — lightweight, experimental

### 5. Multimodal Is the Default

Text-only models are legacy. All major models (GPT-5, Gemini 2.0, Claude Opus 4) handle text, images, audio, video, and code natively. **Gemini Embedding 2** is the first multimodal embedding model — same vector space for text, images, video, and audio.

### 6. Smaller Models + Routing = Cost Efficiency

Instead of throwing everything at the biggest model, the trend is **cooperative model routing**: use small, fast models for 80% of tasks and escalate to large models only when needed. This cuts costs by 60-80%.

### 7. Guardrails & Safety Are Non-Negotiable

Enterprise adoption demands AI governance: content filtering, prompt injection defense, output validation, audit trails, and human-in-the-loop checkpoints. The EU AI Act and similar regulations are now in enforcement.

---

## Part 2: Interview Prep (Key Questions & What to Know)

### Must-Know Concepts

| Topic | One-Liner You Should Be Able to Say |
|-------|--------------------------------------|
| **RAG** | "Retrieval-Augmented Generation grounds LLM responses in external knowledge — retrieve relevant chunks, inject as context, generate answer." |
| **Embeddings** | "Dense vector representations of text that capture semantic meaning. Similar texts have similar vectors." |
| **Vector DB** | "Specialized databases optimized for similarity search on high-dimensional vectors. FAISS for local, Chroma for prototyping, Pinecone for production." |
| **Chunking** | "Splitting documents into smaller pieces for retrieval. RecursiveCharacterTextSplitter at 512 tokens with 50-100 overlap is the benchmark default." |
| **LangChain** | "An orchestration framework that connects LLMs with tools, memory, and data through composable chains using LCEL (LangChain Expression Language)." |
| **LangGraph** | "A stateful agent orchestration framework by LangChain team for production-grade multi-agent systems with cycles, memory, and human-in-the-loop." |
| **Agents** | "LLM + tools + reasoning loop. The agent decides which tool to call and when, using patterns like ReAct (Reason + Act)." |
| **MCP** | "Model Context Protocol — a standard for connecting LLMs to external services. Defines tools, resources, and prompts in a universal format." |
| **LCEL** | "LangChain Expression Language — pipe syntax (prompt | llm | parser) for composing chains declaratively." |
| **Hybrid Search** | "Combining keyword (BM25) and semantic (vector) search. Reciprocal Rank Fusion merges both result sets." |

### Top Interview Questions (with Quick Answers)

**Q1: What is RAG and why use it instead of fine-tuning?**
> RAG retrieves relevant documents at query time and feeds them as context. vs fine-tuning which bakes knowledge into model weights. RAG is cheaper, easier to update (just add documents), doesn't need GPU training, and gives you source attribution. Fine-tune when you need to change the model's *behavior* or *style*, not just its knowledge.

**Q2: How do you evaluate a RAG pipeline?**
> Retrieval quality: Precision@k, Recall@k, F1@k, MRR, NDCG. Generation quality: faithfulness (does the answer match the context?), relevance (does it answer the question?), hallucination rate. Tools like Ragas and DeepEval automate this.

**Q3: Explain the difference between an LLM chain and an agent.**
> A chain follows a fixed sequence (prompt → LLM → parser). An agent has a reasoning loop — it decides which tools to call, observes results, and decides what to do next. Agents are dynamic; chains are static.

**Q4: What chunking strategy would you choose and why?**
> Start with RecursiveCharacterTextSplitter (512 tokens, 50-100 overlap). It preserves document structure by splitting on paragraphs first, then sentences, then words. Tune based on your retrieval evaluation metrics. Semantic chunking can help for knowledge bases but may produce fragments that are too small.

**Q5: How would you handle a query that doesn't need retrieval?**
> Use a router. Classify the query (retrieval vs general) using the LLM, then route accordingly. General questions go straight to the LLM. Domain-specific questions go through the RAG pipeline. Saves cost and latency.

**Q6: What is an embedding and how is it used in search?**
> An embedding maps text to a dense vector (e.g., 768 floats). Similar texts produce similar vectors. For search: embed all documents, store vectors in an index (FAISS/Chroma), embed the query, find nearest vectors via cosine similarity.

**Q7: Compare Chroma, Weaviate, and Pinecone.**
> Chroma: embedded, free, great for prototyping (<1M vectors). Weaviate: open-source with native hybrid search, good for medium scale. Pinecone: fully managed serverless, scales to billions, zero ops but costs $500+/mo. Pick based on scale, budget, and whether you need hybrid search.

**Q8: What is the manager/worker pattern in multi-agent systems?**
> A supervisor agent routes tasks to specialized worker agents. Each worker has focused tools and prompts. The supervisor classifies the request, delegates to the right worker, and combines results. It's more scalable than one agent with 20 tools.

### Resources for Deeper Prep

- [40 GenAI Interview Questions That Actually Get Asked in 2026](https://towardsai.net/p/machine-learning/40-generative-ai-interview-questions-that-actually-get-asked-in-2026-with-answers) — Towards AI
- [30 Agentic AI Interview Questions (Beginner to Advanced)](https://www.analyticsvidhya.com/blog/2026/02/agentic-ai-interview-questions-and-answers/) — Analytics Vidhya
- [Top 30 RAG Interview Questions for 2026](https://www.datacamp.com/blog/rag-interview-questions) — DataCamp
- [Top 30 Agentic AI Interview Questions for 2026](https://www.datacamp.com/blog/agentic-ai-interview-questions) — DataCamp
- [Gen AI Interview Prep Guide](https://www.simplilearn.com/generative-ai-interview-questions-article) — Simplilearn

### Trends to Follow

- [7 Agentic AI Trends to Watch in 2026](https://machinelearningmastery.com/7-agentic-ai-trends-to-watch-in-2026/) — Machine Learning Mastery
- [AI Agent Trends 2026 Report](https://cloud.google.com/resources/content/ai-agent-trends-2026) — Google Cloud
- [Agentic AI Strategy](https://www.deloitte.com/us/en/insights/topics/technology-management/tech-trends/2026/agentic-ai-strategy.html) — Deloitte
- [The Future of GenAI: 10 Trends](https://www.techtarget.com/searchenterpriseai/feature/The-future-of-generative-AI-Trends-to-follow) — TechTarget

---

*Last updated: April 2026 | BIA School of Technology & AI, Pune-Kharadi*
