"""Smart Study Assistant — Streamlit UI
Phase 7: Add a web interface to your CLI project.

Run with: streamlit run app.py
"""
import streamlit as st
import os

# Page config (this one's free — it just sets the browser tab title)
st.set_page_config(
    page_title="Smart Study Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# TODO 22: Create the sidebar ✅
# ============================================================
st.sidebar.title("📚 Smart Study Assistant")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Select Mode:",
    ["💬 Ask (RAG)", "🤖 Agent Mode", "📝 Summarize", "🃏 Flashcards", "❓ Quiz"]
)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📤 Upload study notes", type=["txt"])

st.sidebar.markdown("---")
reindex_button = st.sidebar.button("🔄 Re-index Notes")

st.sidebar.markdown("---")
st.sidebar.caption("Phase 7 · LangChain + Gemini + ChromaDB")


# ============================================================
# TODO 23: Initialize session state ✅
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "agent" not in st.session_state:
    st.session_state.agent = None

if "tools_dict" not in st.session_state:
    st.session_state.tools_dict = None

if "ready" not in st.session_state:
    st.session_state.ready = False


# ============================================================
# TODO 24: Load the study assistant on first run ✅
# ============================================================
@st.cache_resource
def load_study_assistant():
    """Load vectorstore, RAG chain, agent and tools — cached so it runs once."""
    from loader import load_and_chunk
    from vectorstore import load_vectorstore, create_vectorstore
    from retriever import build_rag_chain
    from agent import create_study_agent
    from tools import get_all_tools, summarize_topic, generate_flashcards, quiz_me

    status = st.empty()
    try:
        status.info("⏳ Initializing... Loading your notes and AI models...")

        if os.path.exists("./chroma_db"):
            status.info("📂 Found existing vector store — loading it...")
            vs = load_vectorstore()
        else:
            status.info("📝 Chunking your study notes...")
            chunks = load_and_chunk("data/sample_notes.txt")
            status.info("🔢 Creating embeddings and vector store (first run takes ~30s)...")
            vs = create_vectorstore(chunks)

        status.info("🔗 Building RAG chain...")
        rag_chain = build_rag_chain(vs)

        status.info("🤖 Creating study agent...")
        agent_obj = create_study_agent()

        tools = {
            "summarize": summarize_topic,
            "flashcards": generate_flashcards,
            "quiz": quiz_me,
        }

        status.success("✅ Ready! Your study assistant is loaded.")
        return vs, rag_chain, agent_obj, tools

    except Exception as e:
        status.error(f"❌ Error initializing assistant: {str(e)}")
        st.info("Make sure 'data/sample_notes.txt' exists and your GOOGLE_API_KEY is set in .env")
        return None, None, None, None


if not st.session_state.ready:
    vs, rag_chain, agent_obj, tools = load_study_assistant()
    if vs is not None:
        st.session_state.vectorstore = vs
        st.session_state.rag_chain = rag_chain
        st.session_state.agent = agent_obj
        st.session_state.tools_dict = tools
        st.session_state.ready = True


# ============================================================
# Main Chat Area
# ============================================================
st.title("📚 Smart Study Assistant")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"**Current Mode:** {mode}")
with col2:
    if st.session_state.ready:
        st.success("Ready ✅")
    else:
        st.warning("Loading...")

st.markdown("---")


# ============================================================
# TODO 25: Display chat history ✅
# ============================================================
if st.session_state.messages:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ============================================================
# TODO 26: Handle user input ✅
# ============================================================
user_input = st.chat_input("Ask me anything about your study notes...")

if user_input and st.session_state.ready:
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    answer = ""
    try:
        with st.spinner("🤔 Thinking..."):
            if mode == "💬 Ask (RAG)":
                from evaluator import self_refine
                raw_answer = st.session_state.rag_chain.invoke(user_input)
                answer = self_refine(user_input, raw_answer)

            elif mode == "🤖 Agent Mode":
                from agent import chat_with_agent
                answer = chat_with_agent(st.session_state.agent, user_input)

            elif mode == "📝 Summarize":
                result = st.session_state.tools_dict["summarize"].invoke(user_input)
                answer = result.content if hasattr(result, "content") else str(result)

            elif mode == "🃏 Flashcards":
                result = st.session_state.tools_dict["flashcards"].invoke(user_input)
                answer = result.content if hasattr(result, "content") else str(result)

            elif mode == "❓ Quiz":
                result = st.session_state.tools_dict["quiz"].invoke(user_input)
                answer = result.content if hasattr(result, "content") else str(result)

            else:
                answer = "Mode not recognized."

    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"

    # Display assistant response and save to history
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})


# ============================================================
# TODO 27: Show retrieved sources (RAG mode only) ✅
# ============================================================
if "RAG" in mode and st.session_state.ready:
    with st.expander("📄 Retrieved Sources"):
        try:
            last_user_message = None
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "user":
                    last_user_message = msg["content"]
                    break

            if last_user_message:
                docs = st.session_state.vectorstore.similarity_search(last_user_message, k=3)
                if docs:
                    st.markdown("**Documents used to answer your question:**")
                    for i, doc in enumerate(docs, 1):
                        preview = doc.page_content[:200]
                        if len(doc.page_content) > 200:
                            preview += "..."
                        st.markdown(f"**Document {i}:**\n{preview}")
                        st.markdown("---")
                else:
                    st.info("No relevant documents found for this query.")
            else:
                st.info("No documents retrieved yet. Ask a question first.")
        except Exception as e:
            st.warning(f"Could not retrieve sources: {str(e)}")


# ============================================================
# TODO 28: Handle re-index button ✅
# ============================================================
if reindex_button:
    with st.spinner("🔄 Re-indexing your notes..."):
        try:
            import gc

            # Step 1: Clear the @st.cache_resource cache (drops cached Chroma object)
            load_study_assistant.clear()

            # Step 2: Release all session state references
            st.session_state.vectorstore = None
            st.session_state.rag_chain = None
            st.session_state.agent = None
            st.session_state.tools_dict = None
            st.session_state.ready = False
            st.session_state.messages = []

            # Step 3: Force garbage collection to close any open SQLite handles
            gc.collect()

            # Note: We do NOT delete ./chroma_db here.
            # create_vectorstore() deletes and recreates the collection internally
            # using ChromaDB's own API — no file lock conflicts.

            st.success("✅ Re-index triggered! Reloading...")
            st.rerun()
        except Exception as e:
            st.error(f"Error re-indexing: {str(e)}")


# --- Footer ---
st.markdown("---")
st.caption("Smart Study Assistant • Built with Streamlit + LangChain + Gemini")
