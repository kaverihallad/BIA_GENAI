"""Smart Study Assistant — Streamlit UI
A web-based interface for your AI-powered study assistant.
"""

import streamlit as st
import os
from datetime import datetime

# --- Custom CSS for nice styling ---
st.set_page_config(
    page_title="Smart Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat styling
custom_css = """
<style>
    :root {
        --primary-color: #0d9488;
        --secondary-color: #14b8a6;
        --background: #f0fdfa;
        --text-dark: #0f172a;
    }

    .stChatMessage {
        background-color: white;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid var(--primary-color);
    }

    .user-message {
        background-color: var(--secondary-color);
        color: white;
        border-left-color: var(--primary-color);
    }

    h1, h2, h3 {
        color: var(--primary-color);
    }

    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 6px;
    }

    .stButton > button:hover {
        background-color: var(--secondary-color);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- Sidebar Configuration ---
# TODO 22: Create sidebar with mode selector and file uploader
# - Add st.sidebar.title("Smart Study Assistant")
# - Create mode selector with st.sidebar.radio() - options: "💬 Ask (RAG)", "🤖 Agent Mode", "📝 Summarize", "🃏 Flashcards", "❓ Quiz"
# - Add file uploader with st.sidebar.file_uploader("Upload study notes", type=["txt"])
# - Add a "Re-index" button with st.sidebar.button("🔄 Re-index Notes")
# - Hint: Store the selected mode in a variable (not session state, just local)

st.sidebar.title("📚 Smart Study Assistant")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Select Mode:",
    [
        "💬 Ask (RAG)",
        "🤖 Agent Mode",
        "📝 Summarize",
        "🃏 Flashcards",
        "❓ Quiz"
    ],
    help="Choose how you want to interact with your study notes"
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📤 Upload study notes",
    type=["txt"],
    help="Upload a .txt file containing your study material"
)

reindex_button = st.sidebar.button(
    "🔄 Re-index Notes",
    help="Re-process and index your notes into the vector database"
)

st.sidebar.markdown("---")
st.sidebar.caption("Phase 7: Streamlit UI • BIA Beginner Project")


# --- Initialize Session State ---
# TODO 23: Initialize session state variables
# Create these if they don't exist:
#   - st.session_state.messages = [] (list of dicts with "role" and "content")
#   - st.session_state.vectorstore = None
#   - st.session_state.rag_chain = None
#   - st.session_state.agent = None
#   - st.session_state.tools_dict = None
#   - st.session_state.ready = False (flag to track if everything is loaded)
#
# Use: if "messages" not in st.session_state:
#      st.session_state.messages = []
#
# Hint: Do this for all 6 state variables

def init_session_state():
    """Initialize session state variables on first run."""
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


init_session_state()


# --- Load/Initialize on Startup ---
# TODO 24: Load or create the vectorstore, RAG chain, and agent on startup
# This happens once, when vectorstore is None (first run).
#
# Steps:
# 1. Check if st.session_state.vectorstore is None
# 2. If None:
#    a. Show st.info("Initializing... Loading your notes and AI models...")
#    b. Import: loader, vectorstore, retriever, agent, config
#    c. Check if "./chroma_db" directory exists with os.path.exists()
#    d. If it exists: use vectorstore.load_vectorstore() to load saved embeddings
#    e. If NOT:
#       - Use loader.load_and_chunk("data/sample_notes.txt") to load and chunk notes
#       - Create vectorstore with vectorstore.create_vectorstore(chunks)
#       - (Optional: handle uploaded file here too)
#    f. Build RAG chain: retriever.build_rag_chain(st.session_state.vectorstore)
#    g. Create agent: agent.create_study_agent()
#    h. Get tools dict: tools.get_all_tools() - should be a dict or list
#    i. Set st.session_state.ready = True
#    j. Show st.success("Ready! Your study assistant is loaded.")
#
# Hint: Use try/except to catch errors and show them with st.error()
# Hint: The message "Initializing..." should appear only once

@st.cache_resource
def load_study_assistant():
    """Load and initialize all components (cached for performance)."""
    try:
        # Import modules
        from loader import load_and_chunk
        from vectorstore import load_vectorstore, create_vectorstore
        from retriever import build_rag_chain
        from agent import create_study_agent
        from tools import get_all_tools
        import os

        # Create a placeholder for status updates
        status_placeholder = st.empty()
        status_placeholder.info("⏳ Initializing... Loading your notes and AI models...")

        # Load or create vectorstore
        if os.path.exists("./chroma_db"):
            vs = load_vectorstore()
        else:
            status_placeholder.info("📝 Chunking your study notes...")
            chunks = load_and_chunk("data/sample_notes.txt")
            status_placeholder.info("🔢 Creating embeddings and vector store...")
            vs = create_vectorstore(chunks)

        # Build RAG chain
        status_placeholder.info("🔗 Building RAG chain...")
        rag_chain = build_rag_chain(vs)

        # Create agent
        status_placeholder.info("🤖 Creating study agent...")
        agent_obj = create_study_agent()

        # Get tools
        tools = get_all_tools()

        # Clear status and show success
        status_placeholder.success("✅ Ready! Your study assistant is loaded.")

        return vs, rag_chain, agent_obj, tools

    except Exception as e:
        st.error(f"Error initializing assistant: {str(e)}")
        st.info("Make sure 'data/sample_notes.txt' exists and contains your study material.")
        return None, None, None, None


# Load components on startup
if not st.session_state.ready:
    vs, rag_chain, agent_obj, tools = load_study_assistant()
    if vs is not None:
        st.session_state.vectorstore = vs
        st.session_state.rag_chain = rag_chain
        st.session_state.agent = agent_obj
        st.session_state.tools_dict = tools
        st.session_state.ready = True


# --- Main Chat Area ---
st.title("📚 Smart Study Assistant")
st.markdown(f"**Current Mode:** {mode} | Ready: {'✅' if st.session_state.ready else '⏳'}")
st.markdown("---")


# TODO 25: Display chat history
# Loop through st.session_state.messages and display each message
#
# For each message dict in st.session_state.messages:
#   - Extract message["role"] ("user" or "assistant")
#   - Extract message["content"] (the text)
#   - Use: with st.chat_message(msg["role"]):
#   - Display: st.markdown(msg["content"])
#
# Hint: This shows all previous messages in the conversation

if st.session_state.messages:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# TODO 26: Chat input and message handling
# Create a chat input with st.chat_input("Ask me anything about your study notes...")
# If the user types something (if user_input:):
#   1. Add user message to st.session_state.messages
#   2. Display user message immediately with st.chat_message("user")
#   3. Create a placeholder for the assistant response: st.empty()
#   4. Based on selected mode, process the input:
#      - "💬 Ask (RAG)":
#        - Use: answer = st.session_state.rag_chain.invoke(user_input)
#        - Import evaluator and refine: answer = evaluator.self_refine(user_input, answer)
#      - "🤖 Agent Mode":
#        - Use: answer = agent.chat_with_agent(st.session_state.agent, user_input)
#      - "📝 Summarize":
#        - Use: answer = st.session_state.tools_dict["summarize"].invoke(user_input)
#      - "🃏 Flashcards":
#        - Use: answer = st.session_state.tools_dict["flashcards"].invoke(user_input)
#      - "❓ Quiz":
#        - Use: answer = st.session_state.tools_dict["quiz"].invoke(user_input)
#   5. Wrap processing in st.spinner("Thinking...") while running
#   6. Add assistant message to st.session_state.messages
#   7. Display assistant response in the placeholder
#
# Hint: Extract the answer string carefully (some tools return strings, some return objects)
# Hint: Use try/except to handle errors gracefully
# Hint: Show errors with st.error() in the placeholder

user_input = st.chat_input("Ask me anything about your study notes...")

if user_input and st.session_state.ready:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process input based on mode
    response_placeholder = st.empty()

    try:
        with st.spinner("🤔 Thinking..."):
            if mode == "💬 Ask (RAG)":
                from evaluator import self_refine
                answer = st.session_state.rag_chain.invoke(user_input)
                answer = self_refine(user_input, answer)

            elif mode == "🤖 Agent Mode":
                from agent import chat_with_agent
                answer = chat_with_agent(st.session_state.agent, user_input)

            elif mode == "📝 Summarize":
                answer = st.session_state.tools_dict["summarize"].invoke(user_input)
                # Extract content if it's an object
                if hasattr(answer, "content"):
                    answer = answer.content

            elif mode == "🃏 Flashcards":
                answer = st.session_state.tools_dict["flashcards"].invoke(user_input)
                # Extract content if it's an object
                if hasattr(answer, "content"):
                    answer = answer.content

            elif mode == "❓ Quiz":
                answer = st.session_state.tools_dict["quiz"].invoke(user_input)
                # Extract content if it's an object
                if hasattr(answer, "content"):
                    answer = answer.content

            else:
                answer = "Mode not recognized. Please select a valid mode from the sidebar."

        # Display assistant response
        with response_placeholder.container():
            with st.chat_message("assistant"):
                st.markdown(answer)

        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        with response_placeholder.container():
            st.error(f"Error processing your request: {str(e)}")


# TODO 27: Show retrieved sources (RAG mode only)
# Create an expander that shows the context documents retrieved by the RAG chain
#
# This should only show when mode is "💬 Ask (RAG)" and RAG chain exists
#
# Steps:
#   1. Check if mode == "💬 Ask (RAG)" and st.session_state.rag_chain is not None
#   2. Create an expander: with st.expander("📄 Retrieved Sources"):
#   3. Inside the expander:
#      - Try to get documents using vectorstore similarity search
#      - If available, show each document with st.write() or st.markdown()
#      - Number them (1, 2, 3...)
#      - Show a preview of each chunk
#   4. If no documents found, show: st.info("No documents retrieved yet. Ask a question first.")
#
# Hint: You can retrieve docs using: st.session_state.vectorstore.similarity_search(query)
# Hint: Keep previews to 200 characters so the UI doesn't get cluttered

if mode == "💬 Ask (RAG)" and st.session_state.ready:
    with st.expander("📄 Retrieved Sources"):
        if st.session_state.messages:
            # Get the last user message
            last_user_message = None
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "user":
                    last_user_message = msg["content"]
                    break

            if last_user_message:
                try:
                    # Retrieve relevant documents
                    docs = st.session_state.vectorstore.similarity_search(last_user_message, k=3)

                    if docs:
                        st.markdown("**Documents used to answer your question:**")
                        for i, doc in enumerate(docs, 1):
                            preview = doc.page_content[:200]
                            if len(doc.page_content) > 200:
                                preview += "..."
                            st.markdown(f"**Document {i}:**\n{preview}")
                    else:
                        st.info("No relevant documents found for this query.")

                except Exception as e:
                    st.warning(f"Could not retrieve sources: {str(e)}")
        else:
            st.info("No documents retrieved yet. Ask a question first.")


# TODO 28: Handle the Re-index button
# When the user clicks the "Re-index Notes" button in the sidebar:
#   1. Check if reindex_button is True
#   2. If True:
#      a. Show st.info("Re-indexing your notes...")
#      b. Delete the existing chroma_db directory (use shutil.rmtree() or os.system())
#      c. Reset session state: set vectorstore, rag_chain, agent, tools_dict all to None and ready to False
#      d. Clear messages: st.session_state.messages = []
#      e. Show st.success("Notes re-indexed! Refresh the page or start asking.")
#      f. Use st.rerun() to reload the page and trigger re-initialization
#
# Hint: import shutil for directory removal
# Hint: Use try/except to handle deletion errors
# Hint: st.rerun() restarts the script from the top

if reindex_button:
    with st.spinner("🔄 Re-indexing your notes..."):
        try:
            import shutil

            # Delete existing vectorstore
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")

            # Reset session state
            st.session_state.vectorstore = None
            st.session_state.rag_chain = None
            st.session_state.agent = None
            st.session_state.tools_dict = None
            st.session_state.ready = False
            st.session_state.messages = []

            st.success("✅ Notes re-indexed! The page will reload automatically...")
            st.rerun()

        except Exception as e:
            st.error(f"Error re-indexing: {str(e)}")


# --- Footer ---
st.markdown("---")
st.caption(
    "Smart Study Assistant v1.0 • Built with Streamlit, LangChain, and Google Gemini"
)
