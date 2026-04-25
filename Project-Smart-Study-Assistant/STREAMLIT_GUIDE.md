# Phase 7: Build the Streamlit UI for Smart Study Assistant

**Welcome to Phase 7!** You've built a powerful CLI app with RAG, agents, and tools. Now you're adding a beautiful **web interface** so users don't need the terminal anymore.

This guide will walk you through completing TODOs 22-28 in `app.py`. By the end, you'll have a fully functional web app with multiple modes and real-time interactions.

---

## Prerequisites

Before starting Phase 7, you **must** have completed TODOs 1-21:

- **Phase 1 (TODOs 1-6):** Document loading, chunking, embeddings, vector store ✅
- **Phase 2 (TODOs 7-9):** RAG chain with retrieval and generation ✅
- **Phase 3 (TODOs 10-12, 19-21):** Tools and agent creation ✅
- **Phase 4 (TODOs 13-16):** Self-reflection and evaluation ✅
- **Phase 5 (TODOs 17-18):** Query routing ✅
- **Phase 6:** CLI app working (`main.py` runs correctly) ✅

If you haven't completed these, go back to **BUILD_GUIDE.md** and finish those phases first.

---

## What is Streamlit?

Streamlit is a **Python framework for building web apps without HTML/CSS/JavaScript**. You write Python code, and Streamlit turns it into an interactive web interface.

### How Streamlit Works

Every time a user interacts with your app (clicks a button, types in an input), Streamlit **re-runs the entire script from top to bottom**. This is different from traditional web frameworks where the server waits for client requests.

**Key concept:** Streamlit uses **session state** to remember values between re-runs. Without it, variables would reset every time the script runs.

### Quick Streamlit Functions You'll Use

| Function | What It Does |
|----------|-------------|
| `st.title()` | Show a big heading |
| `st.write()` | Display text or data |
| `st.chat_message()` | Create a chat message bubble |
| `st.chat_input()` | Text input for chat |
| `st.session_state` | Remember variables across runs |
| `st.spinner()` | Show "loading" indicator |
| `st.sidebar.*` | Put UI elements in the left sidebar |
| `st.expander()` | Collapsible section |
| `st.success()`, `st.error()`, `st.info()` | Show colored messages |

---

## Install Streamlit

Before coding, install Streamlit:

```bash
pip install streamlit
```

Verify it works:

```bash
streamlit --version
```

You should see something like: `Streamlit, version 1.28.0`

---

## Phase 7 Overview

You're implementing **8 TODOs** in `app.py`:

| TODO | What You're Building | Time |
|------|-------------------|------|
| 22 | Sidebar with mode selector & file uploader | 5 min |
| 23 | Initialize session state | 3 min |
| 24 | Load/create vectorstore on startup | 10 min |
| 25 | Display chat history | 3 min |
| 26 | Chat input & process user messages | 15 min |
| 27 | Show retrieved sources (RAG mode) | 8 min |
| 28 | Handle re-index button | 7 min |

**Total time:** ~50 minutes

---

## TODO 22: Sidebar with Mode Selector

### What You're Building

The left sidebar contains:
1. A title
2. A radio button to select the study mode (Ask, Agent, Summarize, Flashcards, Quiz)
3. A file uploader for custom study notes
4. A "Re-index" button to refresh the vector store

### Streamlit Prompt

Open Google Antigravity or Claude and paste this:

```
Complete TODO 22 in app.py. Create a Streamlit sidebar with:

1. A title "📚 Smart Study Assistant" using st.sidebar.title()
2. A horizontal line separator using st.sidebar.markdown("---")
3. A mode selector using st.sidebar.radio() with the variable name "mode" 
   and options:
   - "💬 Ask (RAG)"
   - "🤖 Agent Mode"
   - "📝 Summarize"
   - "🃏 Flashcards"
   - "❓ Quiz"
4. Another separator
5. A file uploader using st.sidebar.file_uploader() with:
   - Label: "📤 Upload study notes"
   - Accepted types: ["txt"]
   - Store in variable "uploaded_file"
6. Another separator
7. A button using st.sidebar.button() with:
   - Label: "🔄 Re-index Notes"
   - Store in variable "reindex_button"
8. Final separator and caption

The code structure should be straightforward - just calls to st.sidebar functions in sequence.
Do NOT add any logic yet, just the UI elements.
```

### What to Verify

After completing TODO 22:

✅ Run the app: `streamlit run app.py`
✅ You should see a sidebar on the left with:
   - The title "📚 Smart Study Assistant"
   - A radio button selector (the mode should change when you click options)
   - A file upload box
   - A re-index button
✅ The main page still shows the title "📚 Smart Study Assistant"

**Expected screenshot:**
```
┌─────────────────┬────────────────────────────────┐
│  📚 Smart Study │  📚 Smart Study Assistant      │
│  Assistant      │                                │
│  ───────────    │  Current Mode: 💬 Ask (RAG)   │
│  Select Mode:   │  Ready: ✅                     │
│  ◯ 💬 Ask       │  ────────────────────────────  │
│  ◯ 🤖 Agent    │  (Chat messages appear here)  │
│  ◯ 📝 Summarize │                                │
│  ◯ 🃏 Flashcards│  Ask me anything about your   │
│  ◯ ❓ Quiz      │  study notes...               │
│  ───────────    │                                │
│  📤 Upload      │                                │
│  notes (file)   │                                │
│  ───────────    │                                │
│  [Re-index]     │                                │
│  ───────────    │                                │
│  Phase 7...     │                                │
└─────────────────┴────────────────────────────────┘
```

---

## TODO 23: Initialize Session State

### What You're Building

Session state is how Streamlit remembers data between script re-runs. Without it, your chat history would disappear every time you send a message.

You're creating 6 session state variables:
1. `messages` — list of chat messages
2. `vectorstore` — the ChromaDB instance
3. `rag_chain` — the RAG chain for answering
4. `agent` — the study agent
5. `tools_dict` — the tools the agent can use
6. `ready` — flag to track if initialization is complete

### Streamlit Prompt

```
Complete TODO 23 in app.py. Create a function called init_session_state() that:

1. Checks if "messages" is NOT in st.session_state
   - If not, initialize st.session_state.messages = []
2. Checks if "vectorstore" is NOT in st.session_state
   - If not, initialize st.session_state.vectorstore = None
3. Checks if "rag_chain" is NOT in st.session_state
   - If not, initialize st.session_state.rag_chain = None
4. Checks if "agent" is NOT in st.session_state
   - If not, initialize st.session_state.agent = None
5. Checks if "tools_dict" is NOT in st.session_state
   - If not, initialize st.session_state.tools_dict = None
6. Checks if "ready" is NOT in st.session_state
   - If not, initialize st.session_state.ready = False

Then call init_session_state() immediately after the function definition.

Use the pattern: if "variable_name" not in st.session_state:
```

### What to Verify

After completing TODO 23:

✅ Run the app again: `streamlit run app.py`
✅ The app loads without errors
✅ You won't see any visual change (session state is invisible)
✅ Open browser DevTools → Application → Session Storage (you can see it's tracking state internally)

---

## TODO 24: Load/Initialize on Startup

### What You're Building

When the app starts, it needs to:
1. Check if a vector store already exists (from previous runs)
2. If not, load the sample notes and create one
3. Build the RAG chain
4. Create the study agent
5. Load all tools

This is the "heavy lifting" that only happens once (or when you re-index). It's wrapped in `@st.cache_resource` so it doesn't re-run on every script reload.

### Streamlit Prompt

```
Complete TODO 24 in app.py. Create a cached function called load_study_assistant() that:

1. Use @st.cache_resource decorator (provided in the hints)
2. Inside the function:
   a. Import these modules:
      - from loader import load_and_chunk
      - from vectorstore import load_vectorstore, create_vectorstore
      - from retriever import build_rag_chain
      - from agent import create_study_agent
      - from tools import get_all_tools
      - import os
   
   b. Create a status placeholder: status_placeholder = st.empty()
   c. Show initial message: status_placeholder.info("⏳ Initializing... Loading your notes and AI models...")
   
   d. Check if "./chroma_db" directory exists:
      - If YES: Load it with vectorstore.load_vectorstore()
      - If NO: 
        * Show status: "📝 Chunking your study notes..."
        * Load and chunk notes: chunks = load_and_chunk("data/sample_notes.txt")
        * Show status: "🔢 Creating embeddings and vector store..."
        * Create vectorstore: vs = create_vectorstore(chunks)
   
   e. Show status: "🔗 Building RAG chain..."
   f. Build RAG chain: rag_chain = build_rag_chain(vs)
   
   g. Show status: "🤖 Creating study agent..."
   h. Create agent: agent_obj = create_study_agent()
   
   i. Get all tools: tools = get_all_tools()
   
   j. Clear the status and show success message:
      status_placeholder.success("✅ Ready! Your study assistant is loaded.")
   
   k. Return a tuple: (vs, rag_chain, agent_obj, tools)

3. Add error handling with try/except:
   - If any error occurs, show: st.error(f"Error initializing assistant: {str(e)}")
   - Also show: st.info("Make sure 'data/sample_notes.txt' exists...")
   - Return (None, None, None, None)

4. After the function definition, call it:
   if not st.session_state.ready:
       vs, rag_chain, agent_obj, tools = load_study_assistant()
       if vs is not None:
           st.session_state.vectorstore = vs
           st.session_state.rag_chain = rag_chain
           st.session_state.agent = agent_obj
           st.session_state.tools_dict = tools
           st.session_state.ready = True

The @st.cache_resource decorator is already provided in the code. Use it to wrap the function.
```

### What to Verify

After completing TODO 24:

✅ Run the app: `streamlit run app.py`
✅ You should see status messages:
   ```
   ⏳ Initializing...
   📝 Chunking your study notes...
   🔢 Creating embeddings and vector store...
   🔗 Building RAG chain...
   🤖 Creating study agent...
   ✅ Ready!
   ```
✅ The main page shows "Ready: ✅"
✅ If you refresh the page (F5), it loads much faster (cache working!)
✅ If `data/sample_notes.txt` doesn't exist, you see the error message

**Note:** The initialization happens in the background. The `st.cache_resource` decorator ensures it only runs once per app session.

---

## TODO 25: Display Chat History

### What You're Building

The chat history is a list of dictionaries like:
```python
[
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."},
    {"role": "user", "content": "Explain neural networks"},
    {"role": "assistant", "content": "Neural networks are..."},
]
```

You're looping through this list and displaying each message in a styled chat bubble.

### Streamlit Prompt

```
Complete TODO 25 in app.py. Display the chat history:

1. Check if st.session_state.messages is not empty
2. Loop through each message in st.session_state.messages
3. For each message:
   a. Extract the role: msg["role"] (will be "user" or "assistant")
   b. Extract the content: msg["content"] (the text)
   c. Create a chat message bubble using: with st.chat_message(msg["role"]):
   d. Inside the bubble, display the content: st.markdown(msg["content"])

Use this pattern:
if st.session_state.messages:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
```

### What to Verify

After completing TODO 25:

✅ Run the app: `streamlit run app.py`
✅ You won't see any messages yet (we haven't sent any)
✅ The messages will appear once you complete TODO 26 and send a message
✅ Messages should show in styled bubbles (user on right, assistant on left)

---

## TODO 26: Chat Input & Message Processing

### What You're Building

This is the **core functionality**. When a user types a message:
1. Add it to the chat history
2. Display it immediately
3. Based on the selected mode, process it differently
4. Get the response from the LLM/agent/tools
5. Add the response to history and display it

The 5 modes work differently:
- **Ask (RAG):** Use RAG chain + self-refinement
- **Agent Mode:** Use the LangGraph agent with all tools
- **Summarize:** Call the summarize tool
- **Flashcards:** Call the flashcards tool
- **Quiz:** Call the quiz tool

### Streamlit Prompt (Part A: Setup)

```
Complete TODO 26 Part A in app.py. Set up the chat input:

1. Create a chat input: user_input = st.chat_input("Ask me anything about your study notes...")
2. Check if user_input is not empty AND st.session_state.ready is True:
   if user_input and st.session_state.ready:

3. Inside this block:
   a. Add the user message to history:
      st.session_state.messages.append({"role": "user", "content": user_input})
   
   b. Display the user message immediately:
      with st.chat_message("user"):
          st.markdown(user_input)
   
   c. Create a placeholder for the assistant's response:
      response_placeholder = st.empty()

This sets up the UI. The actual processing comes next.
```

### Streamlit Prompt (Part B: Message Processing)

```
Complete TODO 26 Part B in app.py. Process the user message based on mode:

After the placeholder, wrap everything in a try/except block:
try:
    with st.spinner("🤔 Thinking..."):
        # Mode-specific processing here
        answer = ""
        
        if mode == "💬 Ask (RAG)":
            # Import evaluator (already imported above)
            from evaluator import self_refine
            # Invoke RAG chain
            answer = st.session_state.rag_chain.invoke(user_input)
            # Self-refine the answer
            answer = self_refine(user_input, answer)
        
        elif mode == "🤖 Agent Mode":
            # Import agent functions
            from agent import chat_with_agent
            # Use the agent to process the message
            answer = chat_with_agent(st.session_state.agent, user_input)
        
        elif mode == "📝 Summarize":
            # Use the summarize tool
            answer = st.session_state.tools_dict["summarize"].invoke(user_input)
            # Some tools return objects, extract content if needed
            if hasattr(answer, "content"):
                answer = answer.content
        
        elif mode == "🃏 Flashcards":
            # Use the flashcards tool
            answer = st.session_state.tools_dict["flashcards"].invoke(user_input)
            # Extract content if needed
            if hasattr(answer, "content"):
                answer = answer.content
        
        elif mode == "❓ Quiz":
            # Use the quiz tool
            answer = st.session_state.tools_dict["quiz"].invoke(user_input)
            # Extract content if needed
            if hasattr(answer, "content"):
                answer = answer.content
        
        else:
            answer = "Mode not recognized."

except Exception as e:
    # If something goes wrong, show the error
    with response_placeholder.container():
        st.error(f"Error processing your request: {str(e)}")

After the try/except, display the response:
    with response_placeholder.container():
        with st.chat_message("assistant"):
            st.markdown(answer)
    
    # Add to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
```

### What to Verify

After completing TODO 26:

✅ Run the app: `streamlit run app.py`
✅ Type a message in the chat input box
✅ Press Enter or click the send button
✅ Your message appears in a blue bubble (user)
✅ The app shows "🤔 Thinking..." spinner
✅ The assistant's response appears in a white bubble (assistant)
✅ The messages stay in the chat history (don't disappear on reload)
✅ Try each mode:
   - **Ask (RAG):** Should reference your study notes
   - **Agent Mode:** Should use tools intelligently
   - **Summarize:** Should give a 3-4 bullet summary
   - **Flashcards:** Should generate Q&A pairs
   - **Quiz:** Should create a multiple-choice quiz

**Expected interaction:**

```
User: "What is supervised learning?"
[App shows spinner]
Assistant: "Based on your study notes, supervised learning is a machine learning 
approach where models are trained on labeled data..."

User: "Make me flashcards on this topic"
[App switches to Flashcards mode first]
[App shows spinner]
Assistant: "Q: What is supervised learning?
A: A type of machine learning where...

Q: What are labels?
A: In supervised learning, labels are..."
```

---

## TODO 27: Show Retrieved Sources (RAG Mode Only)

### What You're Building

When users ask a question in RAG mode, you want to show them **which documents** the AI used to answer. This builds trust and lets them verify the sources.

An expander (collapsible section) shows:
1. The top-3 most relevant chunks
2. A preview of each chunk (first 200 characters)
3. A note if no documents were found

### Streamlit Prompt

```
Complete TODO 27 in app.py. Create an expandable "Retrieved Sources" section:

1. Check if mode == "💬 Ask (RAG)" and st.session_state.ready:
2. Create an expander: with st.expander("📄 Retrieved Sources"):
3. Inside the expander:
   a. Check if there are any messages in st.session_state.messages
   b. If there are messages:
      - Find the last user message by looping backwards through messages
      - Check if msg["role"] == "user" and store it in last_user_message
   c. If a last_user_message exists:
      - Use similarity_search to retrieve relevant documents:
        docs = st.session_state.vectorstore.similarity_search(last_user_message, k=3)
      - If docs were found:
        * Show a heading: st.markdown("**Documents used to answer your question:**")
        * Loop through each doc with enumerate(docs, 1) to number them starting at 1
        * For each doc:
          - Extract the text: preview = doc.page_content[:200]
          - If the text is longer than 200 chars, add "..." to the preview
          - Display: st.markdown(f"**Document {i}:**\n{preview}")
      - If no docs were found:
        * Show: st.info("No relevant documents found for this query.")
   d. If no messages yet:
      - Show: st.info("No documents retrieved yet. Ask a question first.")

Wrap everything in try/except to handle errors gracefully:
except Exception as e:
    st.warning(f"Could not retrieve sources: {str(e)}")
```

### What to Verify

After completing TODO 27:

✅ Run the app: `streamlit run app.py`
✅ Ask a question in RAG mode
✅ Click "📄 Retrieved Sources" to expand it
✅ You should see the top 3 document chunks used
✅ Each chunk shows a preview (first 200 characters)
✅ Switch to another mode (like Agent Mode) — the expander should disappear
✅ Switch back to RAG — the expander reappears

**Expected screenshot (expanded):**

```
📄 Retrieved Sources
✓ Expanded

Documents used to answer your question:

Document 1:
Supervised learning is a machine learning approach where we train models on 
labeled data. Each training example has an input (features) and a corresponding 
output (label). The model learns the relationship between inputs...

Document 2:
Examples of supervised learning include linear regression, logistic regression, 
decision trees, neural networks, and support vector machines. These models are 
used for classification and regression tasks...

Document 3:
The key difference between supervised and unsupervised learning is the presence 
of labels. In supervised learning, every training example has a known output...
```

---

## TODO 28: Handle Re-index Button

### What You're Building

The "Re-index" button in the sidebar lets users:
1. Delete the old vector store (./chroma_db folder)
2. Reset all session state variables
3. Clear the chat history
4. Reload the page (which triggers initialization again)

This is useful when the user wants to add new study notes or start fresh.

### Streamlit Prompt

```
Complete TODO 28 in app.py. Handle the re-index button click:

1. Check if reindex_button is True (button was clicked):
   if reindex_button:

2. Inside this block, wrap everything in a spinner:
   with st.spinner("🔄 Re-indexing your notes..."):

3. Inside the try block:
   a. Import shutil: import shutil
   b. Check if "./chroma_db" directory exists:
      if os.path.exists("./chroma_db"):
          shutil.rmtree("./chroma_db")  # Delete the directory
   c. Reset all session state variables to None:
      st.session_state.vectorstore = None
      st.session_state.rag_chain = None
      st.session_state.agent = None
      st.session_state.tools_dict = None
      st.session_state.ready = False
      st.session_state.messages = []
   d. Show success message:
      st.success("✅ Notes re-indexed! The page will reload automatically...")
   e. Reload the page:
      st.rerun()

4. In the except block, show the error:
   except Exception as e:
       st.error(f"Error re-indexing: {str(e)}")
```

### What to Verify

After completing TODO 28:

✅ Run the app: `streamlit run app.py`
✅ Click the "🔄 Re-index Notes" button in the sidebar
✅ You see "🔄 Re-indexing..." spinner
✅ Success message appears: "✅ Notes re-indexed!"
✅ Page reloads automatically
✅ Chat history is cleared
✅ The app re-initializes (you see the initialization messages again)
✅ The chroma_db folder is deleted and recreated fresh

**Note:** The second time through initialization will be slower because it needs to rebuild the embeddings from scratch.

---

## Testing Your Streamlit App

### Full Test Checklist

Run your app with:

```bash
streamlit run app.py
```

Then test each section:

#### Sidebar (TODO 22)
- [ ] Sidebar appears on the left
- [ ] Mode selector works (can click each mode)
- [ ] File uploader appears
- [ ] Re-index button appears

#### Initialization (TODO 24)
- [ ] See initialization messages
- [ ] See "Ready: ✅" on main page
- [ ] Reloading the page is fast (cache working)

#### Chat History (TODO 25-26)
- [ ] Type a message
- [ ] Message appears in blue bubble (user)
- [ ] App shows "Thinking..." spinner
- [ ] Response appears in white bubble (assistant)
- [ ] Messages don't disappear on page reload
- [ ] Try each mode:
  - [ ] Ask (RAG) — answers based on notes
  - [ ] Agent Mode — uses tools intelligently
  - [ ] Summarize — gives 3-4 bullets
  - [ ] Flashcards — generates Q&A pairs
  - [ ] Quiz — creates multiple-choice questions

#### Retrieved Sources (TODO 27)
- [ ] In RAG mode, click "📄 Retrieved Sources"
- [ ] See up to 3 documents listed
- [ ] Each shows a preview of the text
- [ ] In other modes, expander doesn't appear
- [ ] Switch back to RAG, expander reappears

#### Re-index (TODO 28)
- [ ] Click "🔄 Re-index Notes" button
- [ ] See "Re-indexing..." spinner
- [ ] Page reloads
- [ ] Chat history is cleared
- [ ] App shows initialization messages
- [ ] chroma_db folder is recreated

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

**Solution:** Install Streamlit
```bash
pip install streamlit
```

### Issue: "No module named 'config'" or other imports fail

**Solution:** Make sure you're running the app from the project directory:
```bash
cd /path/to/Project-Smart-Study-Assistant
streamlit run app.py
```

### Issue: "Chroma_db not found" error

**Solution:** This is normal on first run. The app creates it. If it persists:
1. Make sure `data/sample_notes.txt` exists
2. Try deleting `chroma_db` folder and running again
3. Check for errors in the initialization message

### Issue: Chat messages disappear when page reloads

**Solution:** This means session state isn't initialized. Check TODO 23 is complete and you're using `st.session_state` correctly.

### Issue: "LLM responses are empty"

**Solution:** 
1. Check your Google API key in `config.py`
2. Try increasing `LLM_TEMPERATURE` from 0 to 0.7
3. Check if you have API quota remaining

### Issue: File upload doesn't work

**Solution:** The uploaded file handling is optional for Phase 7. You can extend it in bonus challenges. For now, the app uses the static `data/sample_notes.txt` file.

### Issue: Button doesn't respond

**Solution:** Remember that Streamlit re-runs the entire script on interaction. Make sure the button is defined in the right place (in the sidebar) and the logic is after the button definition.

### Issue: "st.rerun() not found"

**Solution:** Make sure you're using Streamlit 1.27+. Update with:
```bash
pip install --upgrade streamlit
```

Older versions use `st.experimental_rerun()` instead.

---

## Performance Tips

### 1. Caching (Already Implemented)

Your `load_study_assistant()` is wrapped in `@st.cache_resource`. This means:
- It runs only once per app session
- Subsequent calls return the cached result instantly
- The app loads 100x faster on second run

### 2. Session State (Already Implemented)

Your `st.session_state` stores:
- Chat history — persists across re-runs
- Vectorstore — persists so you don't reload embeddings
- Agent — persists so you don't recreate it

Without this, every interaction would be slow.

### 3. Spinners

The `st.spinner("Thinking...")` shows users the app is working. Without it, they might think it crashed.

---

## Next Steps After Phase 7

Once you've completed the Streamlit UI, here are fun extensions:

### 1. Upload Custom Study Notes

Implement the file upload to let users choose their own notes:

```python
if uploaded_file:
    # Save uploaded file
    with open("data/custom_notes.txt", "w") as f:
        f.write(uploaded_file.read().decode())
    # Re-index if requested
    if st.sidebar.button("Index Custom Notes"):
        # Trigger re-index with new file
```

### 2. Add Conversation Settings

Let users adjust:
- Temperature (more creative vs. more factual)
- Top-K (how many documents to retrieve)
- Refinement iterations (how many times to self-refine)

```python
with st.sidebar.expander("⚙️ Settings"):
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
    top_k = st.slider("Retrieved documents", 1, 10, 3)
```

### 3. Export Conversations

Let users download their chat history as a file:

```python
if st.button("📥 Export Chat"):
    import json
    chat_json = json.dumps(st.session_state.messages, indent=2)
    st.download_button("Download as JSON", chat_json, "chat.json")
```

### 4. Add Analytics

Track:
- Most common questions
- Average response time
- Mode usage statistics

### 5. Multi-User Support

Deploy the app to Streamlit Cloud and let multiple users access it simultaneously.

---

## Deployment (Bonus)

Once you're happy with your app, deploy it for free:

### Option 1: Streamlit Cloud (Easiest)

1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your GitHub repo
4. Streamlit automatically deploys your app
5. Share the public URL

### Option 2: Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

Deploy with:

```bash
docker build -t smart-study-assistant .
docker run -p 8501:8501 smart-study-assistant
```

Your app is now at `http://localhost:8501`

### Option 3: Traditional Server

Deploy to AWS EC2, Heroku, or DigitalOcean using the Docker approach above.

---

## Celebration! 🎉

You've built a **full-stack AI application**:

✅ **Backend:** RAG, agents, tools, self-reflection  
✅ **Frontend:** Beautiful Streamlit UI  
✅ **Deployment:** Ready for production  

This is **real software engineering**, not just Jupyter notebooks. You can:
- Show it to your friends and teachers
- Add it to your portfolio
- Extend it for other use cases
- Deploy it to production

### You've Learned

| Skill | Used Where |
|-------|-----------|
| Document loading & chunking | Phase 1 |
| Vector embeddings | Phase 2 |
| RAG chains | Phase 2-3 |
| LLM tools & agents | Phase 3 |
| Self-reflection | Phase 4 |
| Query routing | Phase 5 |
| Web UI with Streamlit | Phase 7 |

These skills apply to almost every AI product: customer support, research assistants, educational tools, medical diagnosis, legal review, code generation, and more.

---

## FAQ

### Q: Can I customize the colors/styling?

**A:** Yes! The CSS in `app.py` uses CSS variables. Change:
```css
:root {
    --primary-color: #0d9488;  /* Teal */
    --secondary-color: #14b8a6;
}
```
to your favorite colors.

### Q: Can I add more study modes?

**A:** Absolutely! Just add another option to the radio button and handle it in TODO 26.

### Q: How do I handle user authentication?

**A:** For now, the app is single-user. To add authentication, use Streamlit's built-in auth or libraries like `streamlit-authenticator`.

### Q: Can I make it work offline?

**A:** Yes, if you use a local LLM instead of Google Gemini. Replace `ChatGoogleGenerativeAI` with `Ollama` or `LlamaCpp`.

### Q: What if my study notes are very long?

**A:** The chunking in Phase 1 handles this. Notes are split into 500-token chunks with 50-token overlap. Try:
- Increasing `CHUNK_SIZE` for longer context
- Decreasing `TOP_K` for faster retrieval

---

## Final Checklist

Before you're done with Phase 7:

- [ ] All 8 TODOs (22-28) are completed
- [ ] App runs without errors: `streamlit run app.py`
- [ ] You can chat in all 5 modes
- [ ] Chat history persists across reloads
- [ ] Retrieved sources show in RAG mode
- [ ] Re-index button works
- [ ] Code is well-formatted and commented
- [ ] You've tested edge cases (empty notes, long messages, etc.)

**Congratulations! You've completed the Smart Study Assistant project.** 🚀

Next step: Deploy it and show the world what you built!

---

## Resources

- **Streamlit Docs:** https://docs.streamlit.io
- **LangChain Docs:** https://python.langchain.com
- **Google Gemini API:** https://ai.google.dev
- **Chroma Vector DB:** https://docs.trychroma.com

**Questions?** Ask your instructors or check the Streamlit Discord community!

Happy coding! 🎓
