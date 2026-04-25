# Teaching Guide: Smart Study Assistant UI Session
## From CLI to Web App in 60 Minutes

---

## 1. Session Overview

**Objective:** Transform the CLI-based Smart Study Assistant (21 TODOs completed) into a polished web application using Streamlit. Students will add a user-friendly interface while reusing all backend logic.

**Duration:** 2–2.5 hours (includes build time + demos)

**Key Principle:** Streamlit lets you turn Python scripts into interactive web apps with zero HTML/CSS/JavaScript knowledge. For beginners, this is transformative—the same business logic that runs in the CLI can power a web UI with just a few imports.

**Why Streamlit?**
- Instant feedback: re-run on every interaction
- Session state: persist data across re-runs (chat history, settings)
- Rich components: buttons, sliders, chat widgets, file uploads
- Minimal boilerplate: 10 lines of Streamlit can replace 50+ lines of web framework code
- Perfect for AI/data projects: live debugging, fast iteration

---

## 2. Learning Objectives

By the end of this session, students will be able to:

1. **Understand Streamlit fundamentals:**
   - How Streamlit re-runs code on interaction
   - When and why to use `st.session_state`
   - How to structure a Streamlit app (top-to-bottom execution)

2. **Convert a CLI to a web UI:**
   - Replace `input()` calls with Streamlit widgets
   - Preserve existing backend logic (RAG, Agent, Tools modules)
   - Map CLI flows to Streamlit modes/navigation

3. **Manage application state:**
   - Store chat history across re-runs
   - Track user settings (model, chunk size, vector DB)
   - Reset state when needed

4. **Build a professional UI:**
   - Layout with sidebar, columns, expanders
   - Chat-based interface with history
   - Visual feedback (spinners, success/error messages)
   - File upload for bulk indexing

5. **Debug Streamlit apps:**
   - Use `st.write()` to inspect values
   - Understand re-run order
   - Identify state-related bugs

---

## 3. Instructor Pacing Guide

### **Part 1: Streamlit Crash Course (30 minutes)**

**Teach these concepts in order:**

#### 1.1 What is Streamlit?
- "Streamlit is Python code that becomes a web app automatically."
- No HTML, no CSS, no JavaScript—just Python.
- Great for dashboards, data apps, and chatbots.
- Used by thousands of data scientists at startups and enterprises (Uber, Databricks, etc.).

#### 1.2 The Re-Run Model (KEY CONCEPT)
This is the hardest concept for beginners. **Spend 5–7 minutes on this.**

```python
# Streamlit script
st.write("Count:", count)
if st.button("Increment"):
    count += 1
st.write("New count:", count)
```

**What happens when user clicks "Increment":**
1. User clicks button → browser sends event to server
2. Server **re-runs the entire script from top to bottom**
3. On re-run, `st.button()` returns `True` (because it was just clicked)
4. `count` increases
5. Both `st.write()` lines execute again with new values
6. UI updates

**Why this matters:**
- No event listeners (like JavaScript)—script re-runs automatically
- Variables reset each re-run (why you need `st.session_state`)
- Order matters (sidebar setup before main area)

**Live analogy:** "Imagine each click = asking your Python script to run again. The script runs top-to-bottom, fast. That's why it feels interactive."

#### 1.3 Session State (KEY CONCEPT)
**Problem:** Variables reset on re-run. How do we keep chat history?
**Solution:** `st.session_state` — a dictionary that persists across re-runs.

```python
# Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add to it
st.session_state.messages.append({"role": "user", "content": user_input})

# Use it
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
```

**Key points:**
- Initialize in `if` block (only once per session)
- Persist across re-runs—survives even if user closes tab and comes back
- Reset only on `st.rerun()` or browser refresh (Ctrl+Shift+R)
- Perfect for chat history, user settings, toggle states

#### 1.4 Core Components You'll Use
Keep it high-level here; show examples in Part 2.

| Component | Use Case |
|-----------|----------|
| `st.write()` / `st.text()` | Display text |
| `st.text_input()` | Single-line input → returns string |
| `st.file_uploader()` | File input → returns file object |
| `st.button()` | Click action → returns True/False |
| `st.sidebar` | Navigation, settings sidebar |
| `st.columns()` | Multi-column layout |
| `st.expander()` | Collapsible section |
| `st.chat_input()` | Chat message input (lower right) |
| `st.chat_message()` | Display chat bubble |
| `st.spinner()` | Loading indicator while task runs |
| `st.success()` / `st.error()` | Colored feedback messages |

---

### **Part 2: Live Demo — Hello World to Chat in 10 Lines (15 minutes)**

**Do this LIVE in an editor or terminal. Type each step slowly.**

**Goal:** Show students how quick Streamlit is. Build from scratch in 10 minutes.

#### Step 1: Install & Hello World (2 min)
```bash
pip install streamlit
```

```python
# app.py
import streamlit as st
st.write("Hello, BIA!")
```

```bash
streamlit run app.py
```

**Point out:** "That's it. A Python script became a web app."

#### Step 2: Add Input (2 min)
```python
import streamlit as st

name = st.text_input("What's your name?")
st.write(f"Hello, {name}!")
```

**Show:** "Type in the box → page re-runs → greeting updates instantly. No refresh needed."

#### Step 3: Add State & Chat History (4 min)
```python
import streamlit as st

st.title("Mini Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
user_input = st.chat_input("You: ")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    # Fake response
    response = f"Echo: {user_input}"
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
```

**Live actions:**
1. Type in chat box
2. Message appears instantly
3. Type again—both messages stay (state persists)
4. **Key insight:** "Chat history survives because of `st.session_state`. Without it, messages would disappear on re-run."

#### Step 4: Add Spinner (1 min)
```python
if user_input:
    # ... existing code ...
    
    with st.spinner("Thinking..."):
        time.sleep(1)  # Fake delay
        response = f"Echo: {user_input}"
    
    st.success("Done!")
```

**Point out:** "Now students see loading states. Professional touch."

---

### **Part 3: Students Build TODOs 22–28 (45 minutes)**

**Structure:**
- **First 10 min:** Walk-through of project structure. Show them `app.py` skeleton. Explain how it differs from `main.py`.
- **Next 30 min:** Guided independent work. Project guide (TODOs) walks them through each step. Instructor circulates, debugs.
- **Last 5 min:** Quick check-in. Spot any stuck students. Offer mini-fixes.

**What each TODO covers:**

| TODO | Component | Time |
|-----|-----------|------|
| 22 | Sidebar nav + initialize session_state | 5 min |
| 23 | Load vector DB + settings UI | 10 min |
| 24 | Display indexed files | 5 min |
| 25 | Chat history display | 5 min |
| 26 | Chat handler: route to RAG/Agent/Tools | 15 min |
| 27 | Display sources (RAG mode) | 3 min |
| 28 | Re-index button + file uploader | 2 min |

**Antigravity note:** If students use IDE Antigravity extension, they can ask Claude to fill in TODOs. Remind them to **review the code** before running it. Streamlit runs immediately, so broken code is obvious.

**Common issues to watch for:**
1. Forgetting `st.session_state` initialization → chat disappears on re-run
2. Calling expensive function (RAG query) on every re-run → app is slow. Solution: wrap in `if` check.
3. Sidebar not showing → need `st.sidebar` context manager.
4. Chat input fires but nothing happens → likely missing `st.rerun()` after adding to session_state.

---

### **Part 4: Show & Tell (15 minutes)**

**Structure:**
- Call 3–4 volunteers (or rotate through groups)
- Each demos their working UI (2–3 min)
- Ask them to show: sidebar, chat, history persistence, at least one mode switch

**Questions to ask them:**
1. "What was the hardest part of adding Streamlit?"
2. "Why does chat history work?"
3. "What happens when you click 'Re-index'?"

**Celebrate:** "You just turned a CLI tool into a web app. This is deployment-ready. Employers hire for this."

---

### **Part 5: Wrap-Up & Next Steps (15 minutes)**

**Recap:**
- Streamlit re-runs on interaction (state-based, not event-driven)
- `st.session_state` persists data across re-runs
- You kept all your backend logic (RAG, Agent, Tools)—just swapped CLI for UI
- This is a full-stack AI application now

**What they learned:**
- Go from terminal input to web widgets (practical web dev)
- How session state works (applies to Flask, FastAPI, Django too)
- How to structure apps (separation of concerns: backend vs UI)

**What's next (optional extension topics):**
- **Deployment:** Push to Streamlit Cloud (free), Heroku, or Docker
- **Styling:** Custom CSS, themes (dark mode by default)
- **Advanced state:** SQLite persistence (save chat across browser refreshes)
- **Multi-page apps:** `st.Page()` for multi-tab navigation
- **Performance:** Caching with `@st.cache_data` for slow operations

**Closing thought:** "In 2 hours, you've built what would take a web developer days with Flask + HTML + CSS. That's the power of Streamlit. Keep building."

---

## 4. Streamlit Key Concepts (Deep Dive)

### 4.1 The Re-Run Model

**Fundamental rule:** Every interaction (button click, input change, slider move) triggers a complete re-run of the entire script.

**Execution flow:**
```
User action (click button)
    ↓
[Browser sends request to server]
    ↓
[Server re-runs script from line 1 to end]
    ↓
[Generate HTML/UI diff]
    ↓
[Send updated UI to browser]
    ↓
[Browser updates instantly]
```

**Why it matters for beginners:**
- Don't think "event listeners" (JavaScript). Think "re-run the whole script."
- Variable values reset unless you use `st.session_state`.
- Expensive operations (API calls, DB queries) run on every re-run unless cached or conditional.

**Example:**
```python
# DON'T DO THIS (inefficient)
results = expensive_api_call()  # Runs every re-run!
st.write(results)

# DO THIS (conditional)
if "results" not in st.session_state:
    st.session_state.results = expensive_api_call()  # Only first time
st.write(st.session_state.results)
```

### 4.2 Session State

**What it is:** A Python dictionary (`st.session_state`) that persists across re-runs and browser refreshes.

**How it works:**
```python
# First time app loads
if "counter" not in st.session_state:
    st.session_state.counter = 0  # Initialize once

# User clicks button
if st.button("Increment"):
    st.session_state.counter += 1  # Survives re-run

st.write(st.session_state.counter)  # Always shows latest value
```

**Key patterns:**

| Pattern | Purpose |
|---------|---------|
| `if "key" not in st.session_state: st.session_state.key = initial_value` | Initialize once |
| `st.session_state.key = new_value` | Update (survives re-run) |
| `del st.session_state.key` | Clear a key |
| `st.session_state.clear()` | Clear all |

**Chat history example (THE pattern for this project):**
```python
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add new message
st.session_state.messages.append({"role": "user", "content": text})

# Display all
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
```

**Why NOT use plain variables:**
```python
# This BREAKS across re-runs
messages = []
if st.button("Add"):
    messages.append("hello")  # Lost on next re-run!
st.write(messages)  # Always empty

# This WORKS
if "messages" not in st.session_state:
    st.session_state.messages = []
if st.button("Add"):
    st.session_state.messages.append("hello")  # Survives
st.write(st.session_state.messages)  # Always has data
```

### 4.3 Layout & Organization

#### Sidebar
```python
st.sidebar.title("Settings")
mode = st.sidebar.radio("Mode:", ["RAG", "Agent", "Tools"])
chunk_size = st.sidebar.slider("Chunk size:", 100, 1000, 500)
```

**Key:** `st.sidebar.*` creates UI in the left sidebar (separate from main area).

#### Columns (Multi-column layout)
```python
col1, col2, col3 = st.columns(3)

with col1:
    st.write("Left column")

with col2:
    st.write("Middle column")

with col3:
    st.write("Right column")
```

#### Expanders (Collapsible sections)
```python
with st.expander("Advanced Settings"):
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.7)
    top_k = st.number_input("Top K:", 1, 10, 5)
```

**Use case for this project:** Hide advanced settings by default.

#### Containers
```python
container = st.container()
container.write("This goes in a box")
container.button("Button inside box")
```

### 4.4 Chat Components

#### st.chat_input()
```python
user_input = st.chat_input("You: ")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
```

**Behavior:**
- Text input fixed at bottom of page
- Returns string on Enter keypress
- Returns `None` when empty
- Always clears after submission

#### st.chat_message()
```python
with st.chat_message("user"):
    st.write("User said hello")

with st.chat_message("assistant"):
    st.write("Assistant says hi")
```

**Roles:** "user", "assistant", or any string. Built-in icons for standard roles.

**Full chat history pattern:**
```python
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("You: ")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Process input
    response = rag_handler(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()
```

### 4.5 Feedback Components

#### Spinners (loading indicator)
```python
with st.spinner("Indexing documents..."):
    result = expensive_operation()
st.success("Indexing complete!")
```

#### Success / Error messages
```python
st.success("File uploaded!")
st.error("Invalid file format")
st.warning("This action cannot be undone")
st.info("Tip: Use arrow keys to navigate")
```

#### Progress bar
```python
progress_bar = st.progress(0)
for i in range(101):
    progress_bar.progress(i)
    time.sleep(0.01)
```

### 4.6 Input Components

#### Text input
```python
name = st.text_input("Name:", max_chars=50)
```

#### Number input
```python
count = st.number_input("Count:", min_value=1, max_value=100, value=10)
```

#### Slider
```python
age = st.slider("Age:", 0, 120, 25)
```

#### Radio (mutually exclusive)
```python
mode = st.radio("Mode:", ["RAG", "Agent", "Tools"])
```

#### Checkbox
```python
agree = st.checkbox("I agree to terms")
```

#### Selectbox (dropdown)
```python
model = st.selectbox("Model:", ["GPT-4", "Claude", "Llama"])
```

#### File uploader
```python
uploaded_file = st.file_uploader("Upload PDF:", type=["pdf"])
if uploaded_file:
    content = uploaded_file.read()
```

### 4.7 Control Flow

#### st.rerun()
Forces script to re-run immediately.
```python
if st.button("Refresh"):
    st.rerun()  # Script runs again from top
```

#### st.stop()
Halts execution (rest of script skipped).
```python
if not user_logged_in:
    st.write("Please log in")
    st.stop()  # Never reaches code below
st.write("Secret data")
```

#### Conditional rendering
```python
if st.session_state.mode == "RAG":
    display_rag_ui()
elif st.session_state.mode == "Agent":
    display_agent_ui()
```

### 4.8 Caching (Performance optimization)

#### @st.cache_data
Cache results of a function. Re-run only if inputs change.
```python
@st.cache_data
def load_vector_db(path):
    return load_db(path)  # Only runs once, not on every re-run

db = load_vector_db("./vectors.db")
```

**Use case:** Expensive operations (loading large files, API calls).

#### @st.cache_resource
Cache Python objects (like database connections).
```python
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key="...")

client = get_openai_client()  # Same instance every time
```

---

## 5. Common Mistakes & Fixes

### Mistake 1: Chat History Disappears on Re-run
**Problem:**
```python
messages = []
st.chat_input()  # User types message
messages.append(msg)  # Added, then...
st.write(messages)  # On re-run, messages is empty!
```

**Fix:**
```python
if "messages" not in st.session_state:
    st.session_state.messages = []
# Now messages persist
```

---

### Mistake 2: Expensive Operation Runs Every Re-run
**Problem:**
```python
results = openai.ChatCompletion.create(...)  # Runs on every click!
```

**Fix:**
```python
if "results" not in st.session_state:
    st.session_state.results = openai.ChatCompletion.create(...)
st.write(st.session_state.results)
```

---

### Mistake 3: Button Clicked But Nothing Happens
**Problem:**
```python
if st.button("Send"):
    st.session_state.messages.append(msg)
    # Script continues, doesn't show updated chat
```

**Fix:**
```python
if st.button("Send"):
    st.session_state.messages.append(msg)
    st.rerun()  # Force re-run to display new message
```

---

### Mistake 4: Sidebar Not Showing
**Problem:**
```python
st.write("Main")
st.sidebar.write("Sidebar")  # Order doesn't matter here
```

**Fix:** Should work. Check for indentation errors in with block.

---

### Mistake 5: File Uploader Always Shows "Upload Again"
**Problem:**
```python
file = st.file_uploader("Pick file")
if file:
    # Process file
    st.success("Done!")
    # On next re-run, file is None again
```

**Fix:**
```python
file = st.file_uploader("Pick file")
if file and "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = True
    process_file(file)

if st.session_state.get("file_uploaded"):
    st.success("File already processed")
```

---

### Mistake 6: Confusing st.write() with st.text()
**Not really a mistake, but:**
- `st.write(x)` — smart, detects type (str, int, df, etc.) and renders appropriately
- `st.text(x)` — always renders as plain text
- Use `st.write()` 90% of the time.

---

### Mistake 7: Sidebar Slider Value Doesn't Update UI
**Problem:**
```python
chunk_size = st.sidebar.slider("Size:", 100, 1000, 500)
st.write(f"Chunk size: {chunk_size}")
# Slider changes, but display doesn't update
```

**Fix:** This should work by default. If not, check indentation or re-run with Ctrl+Shift+R (hard refresh).

---

## 6. Quick Reference: Streamlit Functions Cheat Sheet

### Display
| Function | Purpose |
|----------|---------|
| `st.write()` | Smart render (detects type) |
| `st.text()` | Plain text |
| `st.markdown()` | Markdown |
| `st.title()` | Large heading |
| `st.header()` | Medium heading |
| `st.subheader()` | Small heading |
| `st.code()` | Code block |
| `st.json()` | JSON object |

### Input
| Function | Returns |
|----------|---------|
| `st.text_input()` | str |
| `st.number_input()` | int/float |
| `st.slider()` | int/float |
| `st.checkbox()` | bool |
| `st.radio()` | selected value |
| `st.selectbox()` | selected value |
| `st.multiselect()` | list of values |
| `st.text_area()` | str (multiline) |
| `st.file_uploader()` | file object |
| `st.chat_input()` | str (or None) |
| `st.button()` | bool (True if clicked) |

### Layout
| Function | Purpose |
|----------|---------|
| `st.sidebar` | Add to sidebar |
| `st.columns()` | Multi-column layout |
| `st.container()` | Group elements |
| `st.expander()` | Collapsible section |
| `st.tabs()` | Tabbed navigation |

### Chat
| Function | Purpose |
|----------|---------|
| `st.chat_input()` | Chat message input |
| `st.chat_message()` | Chat bubble |

### Feedback
| Function | Purpose |
|----------|---------|
| `st.success()` | Green success message |
| `st.error()` | Red error message |
| `st.warning()` | Yellow warning |
| `st.info()` | Blue info |
| `st.spinner()` | Loading indicator |
| `st.progress()` | Progress bar |

### Control
| Function | Purpose |
|----------|---------|
| `st.rerun()` | Force re-run |
| `st.stop()` | Stop execution |
| `st.session_state` | Persistent state dict |

### State Management
| Pattern | Use Case |
|---------|----------|
| `if "key" not in st.session_state: st.session_state.key = value` | Initialize once |
| `st.session_state.key = new_value` | Update |
| `del st.session_state.key` | Delete |
| `st.session_state.clear()` | Clear all |

### Caching
| Decorator | Use Case |
|-----------|----------|
| `@st.cache_data` | Cache function results |
| `@st.cache_resource` | Cache objects (DB connections) |

---

## 7. Session Structure & File Organization

**Project structure after UI completion:**
```
smart-study-assistant/
├── main.py                 # CLI (original, unchanged)
├── app.py                  # Streamlit UI (NEW)
├── rag_handler.py          # RAG logic (reused)
├── agent_handler.py        # Agent logic (reused)
├── tools_handler.py        # Tools logic (reused)
├── vector_db.py            # Vector DB manager (reused)
├── file_processor.py       # Document processing (reused)
├── index/                  # Vector DB storage
│   └── vectors.pkl
├── notes/                  # Sample notes
│   ├── math.txt
│   ├── physics.txt
│   └── ...
└── requirements.txt
```

**app.py structure (what students build):**
```python
import streamlit as st
from rag_handler import RAGHandler
from agent_handler import AgentHandler
from tools_handler import ToolsHandler
from vector_db import VectorDB

# 1. Initialize session state (TODO 22)
if "mode" not in st.session_state:
    st.session_state.mode = "RAG"
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Sidebar (TODO 22-23)
st.sidebar.title("Settings")
st.session_state.mode = st.sidebar.radio("Mode:", ["RAG", "Agent", "Tools"])
chunk_size = st.sidebar.slider("Chunk size:", 100, 1000, 500)

# 3. Load backend (TODO 23-24)
db = VectorDB("./index")
st.write(f"Indexed files: {len(db.list_files())}")

# 4. Display chat history (TODO 25)
st.title("Smart Study Assistant")
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. Chat handler (TODO 26)
user_input = st.chat_input("You: ")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Route to appropriate handler
    if st.session_state.mode == "RAG":
        response = RAGHandler(db, chunk_size).query(user_input)
    elif st.session_state.mode == "Agent":
        response = AgentHandler(db).run(user_input)
    elif st.session_state.mode == "Tools":
        response = ToolsHandler().process(user_input)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# 6. File upload & re-index (TODO 27-28)
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload PDF:")
if uploaded_file:
    with st.spinner("Indexing..."):
        db.add_document(uploaded_file)
    st.sidebar.success("Indexed!")
```

---

## 8. Demo Script (for instructor)

**Use this to show students how to build their first Streamlit app.**

```bash
# 1. Create a new file
touch demo.py

# 2. Type (slowly, narrate as you go):
cat > demo.py << 'EOF'
import streamlit as st
import time

st.title("Chat Demo")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get input
user_input = st.chat_input("You: ")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    # Simulate thinking
    with st.spinner("Thinking..."):
        time.sleep(1)
        response = f"You said: {user_input}"
    
    # Add response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
    st.success("Done!")
EOF

# 3. Run it
streamlit run demo.py

# 4. Test interactivity:
# - Type "hello" → see message
# - Type "world" → both messages stay (session state!)
# - Refresh browser → messages still there (persist across refresh)
```

**Narration while running:**
- "I'm creating a Streamlit app from scratch."
- "Three lines of imports, ten lines of chat logic."
- "Now I run it. Streamlit starts a local server on port 8501."
- "Type in the chat box..."
- "See? Instant feedback. Message appears in chat bubble."
- "Type again. Both messages stay. That's session state working."
- "Now watch: I'll refresh the browser entirely."
- "Messages are still there. They're stored in session_state, so they survive."
- "This is the foundation of your Smart Study Assistant UI."

---

## 9. Troubleshooting Guide

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| "ModuleNotFoundError: No module named 'streamlit'" | Streamlit not installed | `pip install streamlit` |
| Chat history disappears after re-run | Not using `st.session_state` | Initialize all state in `if` block |
| Button clicks don't trigger actions | Not calling `st.rerun()` | Add `st.rerun()` after state change |
| Sidebar empty | `st.sidebar` block not indented | Check indentation of sidebar code |
| File uploader always shows "Upload again" | Not persisting file state | Store filename in `st.session_state` |
| App is slow | Expensive operation on every re-run | Wrap in `if` or use `@st.cache_data` |
| Chat input not visible | Scrolling issue | Use `st.write()` less, `st.chat_message()` more |
| Slider value doesn't update | Rare CSS issue | Refresh with Ctrl+Shift+R |

---

## 10. Assessment Checklist

**By end of session, students should have:**

- [ ] Sidebar with mode selector (RAG/Agent/Tools)
- [ ] Working chat interface with history
- [ ] Chat history persists across re-runs
- [ ] At least one mode wired to backend handler
- [ ] File upload button in sidebar
- [ ] Success/error feedback on interactions
- [ ] All 7 TODOs (22–28) complete

**Stretch goals (if time):**
- [ ] Display sources/reasoning (RAG mode)
- [ ] Settings sliders (chunk size, temperature)
- [ ] File management UI (list indexed docs, delete)

---

## 11. Resources for Students

**Official Streamlit docs:**
- Gallery: https://streamlit.io/gallery
- API Reference: https://docs.streamlit.io
- Cheat Sheet: https://docs.streamlit.io/library/cheatsheet

**Related topics:**
- Session state deep dive: https://docs.streamlit.io/library/api-reference/session-state
- Chat components: https://docs.streamlit.io/library/api-reference/chat
- Caching: https://docs.streamlit.io/library/api-reference/performance

**Deployment:**
- Streamlit Cloud (free): https://streamlit.io/cloud
- Docker: https://docs.streamlit.io/deploy/tutorials/docker

---

## 12. Instructor Notes

**Before session:**
- Install Streamlit locally: `pip install streamlit`
- Test the demo script (5 min)
- Download the project template (if using one)
- Ensure students have latest Python 3.8+

**During session:**
- Keep pacing fast (TODOs are small, finishable)
- Circulate; many students will have `st.session_state` confusion
- If a student gets stuck on TODO 26 (chat handler), help them understand the routing logic, not Streamlit
- Have the demo script handy to re-show how session state works

**Tips for engagement:**
- Celebrate each completed TODO (it's a small win)
- Show real-world Streamlit apps during break (Uber's cost model, Databricks notebooks, etc.)
- Mention that Streamlit is used in production at major companies

**Post-session:**
- Share deployment guide (Streamlit Cloud is free, no credit card)
- Share link to advanced Streamlit tutorial (multi-page apps, caching, themes)
- Assign optional stretch goal: deploy to Streamlit Cloud

---

**End of Teaching Guide**

*Total length: ~15KB. Lighter than a full 2-day course but complete enough to teach and support student projects.*
