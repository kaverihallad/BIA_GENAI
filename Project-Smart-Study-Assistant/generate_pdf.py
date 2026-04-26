"""Generate a PDF document explaining the Smart Study Assistant app."""
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os


class PDF(FPDF):
    def header(self):
        self.set_fill_color(30, 30, 50)
        self.rect(0, 0, 210, 18, "F")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(180, 200, 255)
        self.set_y(5)
        self.cell(0, 8, "Smart Study Assistant  -  Technical Documentation", align="C")
        self.set_text_color(0, 0, 0)
        self.ln(14)

    def footer(self):
        self.set_y(-14)
        self.set_fill_color(30, 30, 50)
        self.rect(0, self.get_y(), 210, 14, "F")
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(180, 200, 255)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    # ── helpers ──────────────────────────────────────────────────────────────
    def section_title(self, text):
        self.ln(4)
        self.set_fill_color(40, 60, 120)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 9, f"  {text}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def sub_title(self, text):
        self.ln(2)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 60, 120)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def body(self, text):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bullet(self, items):
        self.set_font("Helvetica", "", 9)
        for item in items:
            self.cell(6)
            self.cell(4, 5, "-")
            self.multi_cell(0, 5, item)

    def code_block(self, text):
        self.set_fill_color(240, 242, 246)
        self.set_font("Courier", "", 8)
        self.multi_cell(0, 4.5, text, fill=True)
        self.ln(1)

    def two_col_table(self, headers, rows):
        col_w = [55, 130]
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(60, 90, 160)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_w[i], 6, f"  {h}", fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)
        fill = False
        for row in rows:
            self.set_fill_color(235, 240, 255) if fill else self.set_fill_color(255, 255, 255)
            self.set_font("Helvetica", "B", 8)
            x0 = self.get_x()
            y0 = self.get_y()
            self.multi_cell(col_w[0], 5.5, f"  {row[0]}", fill=fill)
            h = self.get_y() - y0
            self.set_xy(x0 + col_w[0], y0)
            self.set_font("Helvetica", "", 8)
            self.multi_cell(col_w[1], 5.5, f"  {row[1]}", fill=fill)
            if self.get_y() < y0 + h:
                self.set_y(y0 + h)
            fill = not fill
        self.ln(2)

    def phase_box(self, number, title, todos, description, color):
        r, g, b = color
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 7, f"  Phase {number}: {title}  [{todos}]", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(245, 247, 255)
        self.set_font("Helvetica", "", 8)
        self.multi_cell(0, 5, f"  {description}", fill=True)
        self.ln(1)


# ─────────────────────────────────────────────────────────────────────────────
pdf = PDF()
pdf.set_margins(15, 22, 15)
pdf.set_auto_page_break(True, margin=18)
pdf.add_page()

# ══════════════════════════════════════════════════════════
#  COVER AREA
# ══════════════════════════════════════════════════════════
pdf.set_fill_color(20, 30, 80)
pdf.rect(0, 18, 210, 70, "F")

pdf.set_y(30)
pdf.set_font("Helvetica", "B", 28)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 12, "Smart Study Assistant", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font("Helvetica", "", 13)
pdf.set_text_color(160, 190, 255)
pdf.cell(0, 8, "AI-Powered Learning with RAG, Agents & Streamlit", align="C",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(120, 150, 220)
pdf.cell(0, 6, "LangChain  -  Gemini 2.5 Flash  -  ChromaDB  -  LangGraph  -  Streamlit",
         align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_y(92)
pdf.set_text_color(0, 0, 0)

# ══════════════════════════════════════════════════════════
#  1. OVERVIEW
# ══════════════════════════════════════════════════════════
pdf.section_title("1. Project Overview")
pdf.body(
    "The Smart Study Assistant is a full-stack AI application that transforms plain text study notes "
    "into an interactive learning experience. Students can ask questions, generate summaries, create "
    "flashcards, and take quizzes -- all powered by Google Gemini and a Retrieval-Augmented Generation "
    "(RAG) pipeline backed by ChromaDB.\n\n"
    "The project was built in seven progressive phases, each introducing a new AI/ML concept from the "
    "BIA GenAI course curriculum. The result is a production-quality Python application with a clean "
    "Streamlit web interface."
)

pdf.sub_title("Key Capabilities")
pdf.bullet([
    "Answer questions grounded in your own study notes (RAG)",
    "Summarize any topic in 3-4 concise bullet points",
    "Generate Q&A flashcard sets for active recall practice",
    "Create multiple-choice quizzes with answer keys",
    "Self-critique and refine its own answers (self-reflection loop)",
    "Intelligently route queries to the right handler without user configuration",
    "Full chat UI with persistent history, mode switching, and source transparency",
])

# ══════════════════════════════════════════════════════════
#  2. TECH STACK
# ══════════════════════════════════════════════════════════
pdf.section_title("2. Technology Stack")
pdf.two_col_table(
    ["Component", "Technology & Purpose"],
    [
        ["LLM", "Google Gemini 2.5 Flash -- fast, capable model for generation and reasoning"],
        ["Embeddings", "models/gemini-embedding-001 -- semantic vector representations of text chunks"],
        ["Vector Store", "ChromaDB (langchain-chroma) -- persistent local vector database for similarity search"],
        ["Orchestration", "LangChain LCEL -- composable pipeline syntax (retriever | prompt | llm | parser)"],
        ["Agent Framework", "LangGraph create_react_agent -- ReAct pattern for tool-calling reasoning"],
        ["Web UI", "Streamlit -- Python-native web framework with chat components and session state"],
        ["API Key Mgmt", "python-dotenv -- loads GOOGLE_API_KEY from .env file securely"],
        ["Text Splitting", "RecursiveCharacterTextSplitter -- 500-char chunks with 50-char overlap"],
    ]
)

# ══════════════════════════════════════════════════════════
#  3. ARCHITECTURE
# ══════════════════════════════════════════════════════════
pdf.section_title("3. System Architecture")
pdf.body("The application follows a modular, layered architecture. Each file is a standalone module with a single responsibility:")

pdf.two_col_table(
    ["File", "Role & Key Functions"],
    [
        ["config.py", "Central configuration: model names, chunk sizes, ChromaDB path, API key loading"],
        ["loader.py", "load_text_file() reads UTF-8 text; chunk_text() splits using RecursiveCharacterTextSplitter"],
        ["vectorstore.py", "get_embeddings(), create_vectorstore(), load_vectorstore() -- manages ChromaDB lifecycle"],
        ["retriever.py", "build_rag_chain() assembles the LCEL pipeline; get_llm() initialises Gemini"],
        ["tools.py", "Three @tool functions: summarize_topic, generate_flashcards, quiz_me"],
        ["agent.py", "create_study_agent() builds the LangGraph ReAct agent; chat_with_agent() runs it"],
        ["evaluator.py", "critique_response(), refine_response(), self_refine() loop; Precision@K, Recall@K"],
        ["router.py", "classify_query() uses LLM to detect intent; route_query() dispatches to handler"],
        ["main.py", "CLI entry point -- loads all modules and runs an interactive command loop"],
        ["app.py", "Streamlit web UI -- sidebar, session state, chat history, multi-mode input handling"],
    ]
)

# ══════════════════════════════════════════════════════════
#  4. DATA FLOW
# ══════════════════════════════════════════════════════════
pdf.section_title("4. Data Flow -- How a Query is Processed")
pdf.body("When a user types a question in the Streamlit UI, the following sequence occurs:")
pdf.code_block(
    "User types query\n"
    "       |\n"
    "       v\n"
    "  [app.py] detects selected mode from sidebar\n"
    "       |\n"
    "       +-- Ask (RAG) --> retriever pulls top-3 chunks from ChromaDB\n"
    "       |                  +-- chunks + question --> Gemini --> answer\n"
    "       |                       +-- self_refine() critiques & improves answer\n"
    "       |\n"
    "       +-- Summarize --> summarize_topic tool --> Gemini --> 3-4 bullets\n"
    "       |\n"
    "       +-- Flashcards --> generate_flashcards tool --> Gemini --> 5 Q&A pairs\n"
    "       |\n"
    "       +-- Quiz --> quiz_me tool --> Gemini --> 3 MCQ with answers\n"
    "       |\n"
    "       +-- Agent Mode --> LangGraph ReAct agent decides which tool to call\n"
    "                           +-- tool result returned to user"
)

# ══════════════════════════════════════════════════════════
#  5. BUILD PHASES
# ══════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title("5. Build Phases -- 28 TODOs Across 7 Phases")

phases = [
    (1, "Knowledge Base", "TODOs 1-6  (~30 min)", (40, 100, 60),
     "loader.py & vectorstore.py. Load UTF-8 text files, split into 500-char overlapping chunks using "
     "RecursiveCharacterTextSplitter, generate semantic embeddings via Google's gemini-embedding-001 model, "
     "and persist them in a ChromaDB vector store on disk."),
    (2, "RAG Chain", "TODOs 7-9  (~20 min)", (40, 80, 150),
     "retriever.py. Create a similarity-search retriever (top-K=3), build a grounded prompt template "
     "(answer ONLY from context), and wire everything into a single LCEL expression chain: "
     "{context: retriever | format_docs, question: passthrough} | prompt | llm | StrOutputParser()."),
    (3, "Tools & Agent", "TODOs 10-12, 19-21  (~25 min)", (100, 50, 130),
     "tools.py & agent.py. Implement three @tool-decorated functions (summarize, flashcards, quiz), "
     "then create a LangGraph ReAct agent that reasons about which tool to use based on the user's request."),
    (4, "Self-Reflection", "TODOs 13-16  (~20 min)", (140, 80, 30),
     "evaluator.py. Implement critique_response() (LLM grades its own answer for accuracy/clarity), "
     "refine_response() (LLM improves based on critique), and the self_refine() loop. "
     "Also implements Precision@K and Recall@K retrieval quality metrics."),
    (5, "Query Router", "TODOs 17-18  (~15 min)", (30, 110, 120),
     "router.py. classify_query() sends the user's question to the LLM and gets back exactly one of five "
     "intent labels. route_query() dispatches to the correct handler -- RAG for study questions, "
     "tools for summarize/flashcards/quiz, direct LLM for general questions."),
    (6, "CLI App", "main.py  (~10 min)", (80, 80, 80),
     "main.py is the command-line entry point. It loads the vectorstore, builds the RAG chain, "
     "creates the agent, then runs an interactive loop supporting: ask, agent, summarize, flashcards, quiz, "
     "eval, index, help, and quit commands."),
    (7, "Streamlit UI", "TODOs 22-28  (~50 min)", (20, 80, 140),
     "app.py. Full web interface: sidebar with 5-mode selector, file uploader, and re-index button; "
     "session state for chat persistence; @st.cache_resource startup loader; chat message display; "
     "multi-mode input handler with spinner; Retrieved Sources expander (RAG mode); safe re-index flow."),
]

for num, title, todos, color, desc in phases:
    pdf.phase_box(num, title, todos, desc, color)

# ══════════════════════════════════════════════════════════
#  6. STREAMLIT UI GUIDE
# ══════════════════════════════════════════════════════════
pdf.section_title("6. Streamlit UI -- Features & Modes")

pdf.sub_title("Sidebar Controls")
pdf.bullet([
    "Mode Selector (radio): Switch between Ask (RAG), Agent Mode, Summarize, Flashcards, Quiz",
    "Upload Study Notes: Replace the default sample_notes.txt with your own .txt file",
    "Re-index Notes: Clears and rebuilds the ChromaDB vector store from the current notes file",
])

pdf.sub_title("Main Chat Area")
pdf.bullet([
    "Displays current mode and a Ready / Loading... indicator",
    "Full chat history preserved across page interactions via st.session_state",
    "User messages shown in blue bubbles; assistant responses in white bubbles",
    "Thinking... spinner shown while the LLM / agent is processing",
])

pdf.sub_title("Retrieved Sources Expander (RAG mode only)")
pdf.body(
    "In Ask (RAG) mode, a collapsible 'Retrieved Sources' panel shows the top-3 document chunks "
    "retrieved from ChromaDB that were used to generate the answer. This provides full transparency "
    "and lets students verify the source of each response."
)

# ══════════════════════════════════════════════════════════
#  7. EXAMPLE INTERACTIONS
# ══════════════════════════════════════════════════════════
pdf.section_title("7. Example Interactions")

pdf.two_col_table(
    ["Mode / Input", "Expected Behaviour"],
    [
        ["Ask (RAG)\n\"What % of ML is supervised?\"",
         "Retrieves the relevant chunk from notes, answers with 70%, self-refines the response for clarity"],
        ["Ask (RAG)\n\"Difference between classification & regression?\"",
         "Grounds answer in study notes; sources panel shows the 3 retrieved chunks used"],
        ["Summarize\n\"neural networks\"",
         "Returns 3-4 bullet-point summary of neural networks using Gemini's world knowledge"],
        ["Flashcards\n\"machine learning basics\"",
         "Generates 5 Q&A pairs formatted as Q: ... / A: ... for active recall study"],
        ["Quiz\n\"reinforcement learning\"",
         "Creates a 3-question multiple-choice quiz with A/B/C/D options and the correct answer marked"],
        ["Agent Mode\n\"Make flashcards for my ML exam\"",
         "LangGraph ReAct agent reasons about the request and calls generate_flashcards tool automatically"],
        ["Agent Mode\n\"Quiz me on supervised learning\"",
         "Agent detects quiz intent and calls quiz_me tool, returning a full MCQ set"],
        ["Auto-routed\n\"Summarize deep learning\"",
         "Router classifies as 'summarize', dispatches to summarize_topic tool without manual mode switch"],
    ]
)

# ══════════════════════════════════════════════════════════
#  8. SETUP & RUNNING
# ══════════════════════════════════════════════════════════
pdf.section_title("8. Setup & Running the App")

pdf.sub_title("Prerequisites")
pdf.bullet([
    "Python 3.10 or higher",
    "Google API key from https://aistudio.google.com/app/apikeys (free tier available)",
    "macOS / Linux / Windows with pip and venv",
])

pdf.sub_title("Installation")
pdf.code_block(
    "# 1. Create and activate virtual environment\n"
    "python3 -m venv venv\n"
    "source venv/bin/activate          # macOS/Linux\n"
    "# venv\\Scripts\\activate           # Windows\n\n"
    "# 2. Install dependencies\n"
    "pip install -r requirements.txt\n\n"
    "# 3. Set your API key\n"
    "echo 'GOOGLE_API_KEY=your-key-here' > .env\n\n"
    "# 4. Add study notes\n"
    "# Place your notes in data/sample_notes.txt\n\n"
    "# 5. Run the Streamlit app\n"
    "streamlit run app.py\n\n"
    "# -- OR -- run the CLI version\n"
    "python main.py"
)

pdf.sub_title("Configuration (config.py)")
pdf.two_col_table(
    ["Setting", "Default / Description"],
    [
        ["LLM_MODEL", "gemini-2.5-flash -- the Gemini model used for all generation"],
        ["EMBEDDING_MODEL", "models/gemini-embedding-001 -- embedding model for ChromaDB"],
        ["LLM_TEMPERATURE", "0 -- deterministic responses; set to 0.7 for more creativity"],
        ["CHUNK_SIZE", "500 characters per chunk"],
        ["CHUNK_OVERLAP", "50 characters overlap between adjacent chunks"],
        ["TOP_K", "3 -- number of chunks retrieved per query"],
        ["CHROMA_PERSIST_DIR", "./chroma_db -- local directory for persistent vector store"],
    ]
)

# ══════════════════════════════════════════════════════════
#  9. COURSE CONCEPTS MAP
# ══════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title("9. Course Concepts Applied")
pdf.body("Every component of this project maps directly to a session from the BIA GenAI course curriculum:")

pdf.two_col_table(
    ["Concept Used", "Course Session  ->  Implementation in This Project"],
    [
        ["Document Chunking", "Session 12 -> RecursiveCharacterTextSplitter in loader.py"],
        ["Embeddings", "Sessions 10-11 -> GoogleGenerativeAIEmbeddings in vectorstore.py"],
        ["Vector Store / ChromaDB", "Sessions 10-11 -> Chroma.from_texts() + similarity_search in vectorstore.py"],
        ["RAG with LCEL", "Session 12 -> Full LCEL chain in retriever.py (retriever | prompt | llm | parser)"],
        ["Custom @tool Functions", "Session 9 -> summarize_topic, generate_flashcards, quiz_me in tools.py"],
        ["ReAct Agent", "Session 9 -> create_react_agent (LangGraph) in agent.py"],
        ["Self-Reflection / Self-Refine", "Session 8 -> critique_response + refine_response loop in evaluator.py"],
        ["Retrieval Metrics", "Session 12 -> precision_at_k, recall_at_k, f1_at_k in evaluator.py"],
        ["Query Routing", "Sessions 9+12 -> classify_query + route_query in router.py"],
        ["Streamlit UI", "Phase 7 -> Full web app with chat, session state, caching in app.py"],
    ]
)

# ══════════════════════════════════════════════════════════
#  10. TROUBLESHOOTING
# ══════════════════════════════════════════════════════════
pdf.section_title("10. Troubleshooting")
pdf.two_col_table(
    ["Error", "Fix"],
    [
        ["externally-managed-environment\n(pip install blocked)",
         "Use a virtual environment: python3 -m venv venv && source venv/bin/activate"],
        ["GoogleGenerativeAIError: API key not set",
         "Create a .env file with GOOGLE_API_KEY=your-key and ensure load_dotenv() is called in config.py"],
        ["404 NOT_FOUND -- embedding model",
         "Update EMBEDDING_MODEL in config.py to 'models/gemini-embedding-001'"],
        ["attempt to write a readonly database",
         "Click Re-index Notes; the app now safely deletes the ChromaDB collection internally "
         "using ChromaDB's own API -- no OS-level file deletion, no SQLite lock conflicts"],
        ["No chunks found / vectorstore empty",
         "Verify data/sample_notes.txt exists and has content. Delete ./chroma_db folder and re-index."],
        ["Agent returns no tools",
         "Ensure get_all_tools() in tools.py returns [summarize_topic, generate_flashcards, quiz_me]"],
        ["LLM responses empty or truncated",
         "Increase LLM_TEMPERATURE in config.py from 0 to 0.7 for more varied responses"],
        ["st.rerun() not found",
         "Upgrade Streamlit: pip install --upgrade streamlit (requires v1.27+)"],
    ]
)

# ══════════════════════════════════════════════════════════
#  FOOTER NOTE
# ══════════════════════════════════════════════════════════
pdf.ln(6)
pdf.set_fill_color(240, 245, 255)
pdf.set_font("Helvetica", "I", 8)
pdf.set_text_color(80, 80, 120)
pdf.multi_cell(0, 5,
    "Built as a capstone project for the BIA GenAI course. "
    "Stack: Python 3.10+ - LangChain - LangGraph - Google Gemini 2.5 Flash - "
    "ChromaDB - Streamlit - python-dotenv",
    fill=True
)

out_path = "Smart_Study_Assistant_Documentation.pdf"
pdf.output(out_path)
print(f"PDF saved: {os.path.abspath(out_path)}")
