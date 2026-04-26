# LangChain Patterns Cheat Sheet
## Session 15: LCEL & Runnable API Quick Reference

**Target:** BIA Beginner Students | **Time:** 30-45 min | **Prerequisite:** Smart Study Assistant (retriever.py)

---

## What You Already Know (Quick Recap)

You've been using LCEL this whole time. Your `retriever.py` chain looks like this:

```python
{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
```

**What's happening here:**
- `|` (pipe) = "pass output to next step"
- `{...}` = RunnableParallel (runs multiple things, creates a dict)
- `retriever | format_docs` = pipe retriever output into formatter
- `RunnablePassthrough()` = "just pass the input through unchanged"
- Final result = dict with "context" and "question" keys

**The mental model:** Each step takes input → does work → passes output to next step.

---

## Pattern 1: RunnableParallel — Do Multiple Things at Once

**Problem:** Your app needs both a summary AND flashcards from the same input. Right now you'd call two separate chains and merge the results manually.

**Solution:** `RunnableParallel` runs multiple chains at the same time.

### Simple Example

```python
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Define your chains
summary_chain = prompt_summary | llm | StrOutputParser()
flashcards_chain = prompt_flashcards | llm | StrOutputParser()

# Combine them in parallel
parallel_output = RunnableParallel(
    summary=summary_chain,
    flashcards=flashcards_chain,
)

result = parallel_output.invoke("machine learning")
print(result["summary"])      # The summary text
print(result["flashcards"])   # The flashcards text
```

### Why This Matters

**You already did this!** Your RAG chain uses `RunnableParallel`:

```python
{"context": retriever | format_docs, "question": RunnablePassthrough()}
```

The curly braces `{...}` create a `RunnableParallel` that:
- Runs `retriever | format_docs` to get context
- Runs `RunnablePassthrough()` to pass the question through
- Combines them into a dict `{"context": "...", "question": "..."}`

**Aha moment:** You've been composing chains in parallel without explicitly using `RunnableParallel`!

### Real-World Use

```python
# Quiz study app: get summary, then generate practice questions
study_output = RunnableParallel(
    summary=summary_chain,
    questions=question_generator_chain,
).invoke(study_material)

# Both run in parallel, faster than calling them sequentially
```

---

## Pattern 2: RunnableBranch — If/Else in a Chain

**Problem:** Different types of requests need different processing. You manually wrote if/elif logic in `router.py` to handle this.

**Solution:** `RunnableBranch` is the LangChain-native way to route inputs.

### Simple Example

```python
from langchain_core.runnables import RunnableBranch

# Define your chains
quiz_chain = prompt_quiz | llm | StrOutputParser()
summary_chain = prompt_summary | llm | StrOutputParser()
default_chain = prompt_generic | llm | StrOutputParser()

# Create branching logic
router = RunnableBranch(
    (lambda x: "quiz" in x["type"], quiz_chain),
    (lambda x: "summary" in x["type"], summary_chain),
    default_chain,  # fallback if no conditions match
)

# Invoke with input
result = router.invoke({"type": "quiz", "topic": "biology"})
# Uses quiz_chain because "quiz" is in the type
```

### How It Works

1. **Conditions** are tested top-to-bottom (lambda functions)
2. **First match wins** — if "quiz" is in type, quiz_chain runs
3. **Default fallback** — if no conditions match, use the last chain
4. **Output** — just the result (not a dict like RunnableParallel)

### Before & After

**Your old code (manual routing):**
```python
# router.py style
def route_request(request_type, topic):
    if "quiz" in request_type:
        return generate_quiz(topic)
    elif "summary" in request_type:
        return summarize_topic(topic)
    else:
        return get_generic_info(topic)

# Hard to test, hard to compose, mixes control flow with LLM logic
```

**With RunnableBranch:**
```python
# Clean, composable, testable chains
router = RunnableBranch(
    (lambda x: "quiz" in x["type"], quiz_chain),
    (lambda x: "summary" in x["type"], summary_chain),
    generic_chain,
)

# Now you can pipe it: user_input | router | output_parser
```

### Real-World Use

```python
# Adaptive study app
input_data = {"type": "quiz", "difficulty": "hard", "subject": "calculus"}

chain = (
    RunnableLambda(validate_input) 
    | router  # This routes based on type
    | StrOutputParser()
)

result = chain.invoke(input_data)
```

---

## Pattern 3: RunnableLambda — Any Function as a Chain Step

**Problem:** You need custom Python logic between chain steps. Your `format_docs` function in `retriever.py` is processing data, but it's not technically a "Runnable."

**Solution:** Wrap any Python function with `RunnableLambda` to use it in a chain.

### Simple Example

```python
from langchain_core.runnables import RunnableLambda

def clean_and_lowercase(text):
    """Custom processing function"""
    return text.strip().lower()

# Make it a Runnable
clean = RunnableLambda(clean_and_lowercase)

# Now you can pipe it in a chain
chain = clean | prompt | llm | StrOutputParser()

result = chain.invoke("  HELLO WORLD  ")
# Output: prompt receives "hello world"
```

### Your `format_docs` is Already a Lambda

In `retriever.py`, you have something like:

```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# You could explicitly wrap it:
from langchain_core.runnables import RunnableLambda
format_docs = RunnableLambda(format_docs)

# Or just use it directly in pipes (LangChain handles this)
retriever | format_docs | prompt | llm
```

### Real-World Use

```python
from langchain_core.runnables import RunnableLambda

def extract_json_fields(text):
    """Extract specific fields from LLM output"""
    import json
    try:
        return json.loads(text)
    except:
        return {"error": "Could not parse"}

# Multi-step chain with custom processing
chain = (
    prompt | llm | StrOutputParser() 
    | RunnableLambda(extract_json_fields)  # Custom function in the middle
    | RunnableLambda(lambda x: x.get("answer", "N/A"))
)

result = chain.invoke("What is 2+2?")
# Extracts JSON, then pulls the "answer" field
```

---

## Pattern 4: RunnablePassthrough.assign() — Add Fields Mid-Chain

**Problem:** You want to keep the original input AND add new computed fields. Your RAG chain passes the question through AND adds context — this is `.assign()` in action.

**Solution:** `assign()` merges new fields with existing input.

### Simple Example

```python
from langchain_core.runnables import RunnablePassthrough

# Simple: add a computed field
chain = (
    RunnablePassthrough.assign(
        context=retriever | format_docs  # Compute new field
    )
    | prompt  # Now prompt receives {"question": "...", "context": "..."}
    | llm
    | StrOutputParser()
)

result = chain.invoke("What is machine learning?")
# Input {"question": "What is machine learning?"}
# After assign: {"question": "...", "context": "..."}
# Passed to prompt
```

### How It's Better

**Before (manual dict creation):**
```python
{"context": retriever | format_docs, "question": RunnablePassthrough()}
```

**After (using assign):**
```python
RunnablePassthrough.assign(
    context=retriever | format_docs
)
# Functionally the same, but clearer intent
```

### Real-World Use

```python
from langchain_core.runnables import RunnablePassthrough

# Multi-step refinement
chain = (
    RunnablePassthrough.assign(
        context=retriever | format_docs,
        metadata=RunnableLambda(lambda x: get_metadata(x["topic"]))
    )
    | prompt_with_context  # Receives: {"topic": "...", "context": "...", "metadata": "..."}
    | llm
    | StrOutputParser()
)

result = chain.invoke({"topic": "biology", "level": "beginner"})
# All original fields + computed context + metadata
```

---

## Pattern 5: Fallbacks & Retries — Production Safety

**Problem:** Your app calls Gemini API. Sometimes it's slow, rate-limited, or temporarily down. You need graceful degradation.

**Solution:** Chain methods for handling failures.

### Fallbacks: Try a Backup Chain

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# Primary and backup chains
primary = prompt | ChatGoogleGenerativeAI(model="gemini-2.5-flash") | parser
backup = prompt | ChatGoogleGenerativeAI(model="gemini-1.5-flash") | parser

# Add fallback
safe_chain = primary.with_fallbacks([backup])

result = safe_chain.invoke("Complex question")
# If primary fails, automatically try backup
```

### Retries: Auto-Retry on Errors

```python
from langchain_core.runnables import RunnableRetry

# Retry up to 3 times on failure
chain = (
    prompt 
    | llm.with_retry(
        stop=lambda x: x > 2,  # Stop after 3 attempts
        retry_if=lambda e: "rate_limit" in str(e)  # Only retry on rate limits
    )
    | parser
)

result = chain.invoke("Question")
# Automatically retries transient failures
```

### Real Scenario: Study App Resilience

```python
# Production RAG chain for study app
study_chain = (
    RunnablePassthrough.assign(
        context=retriever | format_docs
    )
    | prompt
    | ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    | StrOutputParser()
).with_fallbacks([
    simple_prompt | ChatGoogleGenerativeAI(model="gemini-1.5-flash") | StrOutputParser()
]).with_retry(
    stop_after_attempt=2
)

# If Gemini-2.5 fails → tries Gemini-1.5
# If both timeout → retries once more
# If still fails → exception (you can catch and show user)
```

---

## Pattern 6: Pipeline Refactoring — Before & After

**Problem:** Your code has RAG logic mixed with string processing, conditionals, and LLM calls all in one function. It's hard to test, reuse, and understand.

**Solution:** Refactor into clean, composable LCEL chains.

### Messy "Before" Code

```python
def study_helper(topic, difficulty, include_quiz):
    """Everything in one function — hard to test or reuse"""
    
    # Custom preprocessing
    topic = topic.strip().lower()
    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"
    
    # Retrieve docs
    docs = retriever.invoke(topic)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Manual prompt construction
    if include_quiz:
        prompt_text = f"""
        Teach about {topic} at {difficulty} level.
        Include a quiz at the end.
        
        Context:
        {context}
        """
    else:
        prompt_text = f"""
        Teach about {topic} at {difficulty} level.
        
        Context:
        {context}
        """
    
    # LLM call
    response = llm.invoke(prompt_text)
    return response.content
```

**Problems:**
- All logic mixed together
- Can't test preprocessing separately
- Can't reuse the routing logic
- Can't parallelize retrieval + validation
- String interpolation instead of proper prompts

### Clean "After" Code

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Validation as a chain step
validate_input = RunnableLambda(
    lambda x: {
        **x,
        "topic": x["topic"].strip().lower(),
        "difficulty": x["difficulty"] if x["difficulty"] in ["easy", "medium", "hard"] else "medium"
    }
)

# 2. Retrieval + context formatting
get_context = retriever | RunnableLambda(
    lambda docs: "\n\n".join(doc.page_content for doc in docs)
)

# 3. Routing based on quiz requirement
quiz_prompt = PromptTemplate.from_template(
    "Teach about {topic} at {difficulty} level. Include a quiz.\n\nContext:\n{context}"
)
no_quiz_prompt = PromptTemplate.from_template(
    "Teach about {topic} at {difficulty} level.\n\nContext:\n{context}"
)

route_prompt = RunnableBranch(
    (lambda x: x.get("include_quiz"), quiz_prompt),
    no_quiz_prompt
)

# 4. Complete pipeline
study_chain = (
    validate_input
    | RunnablePassthrough.assign(context=get_context)
    | route_prompt
    | ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    | StrOutputParser()
)

# 5. Usage — same function call, cleaner internals
result = study_chain.invoke({
    "topic": "  machine learning  ",
    "difficulty": "advanced",
    "include_quiz": True
})
```

**Benefits:**
- **Testable:** Test validation, routing, and retrieval separately
- **Reusable:** `get_context` can be used in other chains
- **Debuggable:** Add `.with_fallbacks()` or retry logic easily
- **Maintainable:** Clear separation of concerns
- **Composable:** Combine with other chains later

### The Refactoring Principle

```
Messy → Pass data through functions manually
        ↓
Clean → Create composable Runnable steps with clear inputs/outputs
        ↓
Result → Easier to test, debug, extend, parallelize
```

---

## Quick Reference Table

| Pattern | Import | Use When | Returns |
|---------|--------|----------|---------|
| **RunnableParallel** | `langchain_core.runnables` | Multiple operations on same input (run in parallel) | Dict with named outputs |
| **RunnableBranch** | `langchain_core.runnables` | Route to different chains based on conditions | Output from chosen branch chain |
| **RunnableLambda** | `langchain_core.runnables` | Wrap custom Python functions for use in chains | Function return value |
| **.assign()** | `RunnablePassthrough` | Add computed fields while keeping original input | Dict with original + new fields |
| **.with_fallbacks()** | Any chain method | Handle failures gracefully (try backup chain) | Output from first successful chain |
| **.with_retry()** | Any chain method | Auto-retry on transient errors | Output from successful attempt |

---

## Common Gotchas & Tips

### Gotcha 1: `.assign()` vs Plain Dict

```python
# This works but less clear
{"context": retriever | format_docs, "question": RunnablePassthrough()}

# This is clearer intent
RunnablePassthrough.assign(context=retriever | format_docs)
```

### Gotcha 2: RunnableBranch Stops at First Match

```python
# Only the first matching condition runs
router = RunnableBranch(
    (lambda x: "quiz" in x.get("type", ""), quiz_chain),  # This one wins
    (lambda x: True, generic_chain),  # Never reached if first matches
)
```

### Gotcha 3: Lambda Functions Capture Values

```python
# WRONG - all conditions use the last value of i
chains = [chain1, chain2, chain3]
router = RunnableBranch(
    *[(lambda x, i=i: x["id"] == i, chains[i]) for i in range(3)],
    default_chain
)
# Correct pattern shown above uses explicit condition functions
```

### Tip: Chain Debugging

```python
# Add intermediate outputs to debug
chain = (
    validate_input
    | RunnableLambda(lambda x: print(f"After validation: {x}") or x)
    | RunnablePassthrough.assign(context=get_context)
    | RunnableLambda(lambda x: print(f"Before prompt: {x}") or x)
    | prompt
    | llm
)
```

---

## Interview Corner

### Q1: What is LCEL and why use it over regular function calls?

**Answer:** LCEL (LangChain Expression Language) is a declarative way to chain LLM operations using the pipe operator `|`. Instead of:

```python
# Regular approach - manual steps
def my_chain(input):
    result1 = retriever(input)
    result2 = format(result1)
    result3 = prompt.format(result2)
    result4 = llm(result3)
    return parser(result4)
```

With LCEL:
```python
chain = retriever | format_fn | prompt | llm | parser
result = chain.invoke(input)
```

**Why it's better:**
- **Automatic parallelization** — LangChain can run parallel steps simultaneously
- **Built-in retry/fallback logic** — `.with_retry()`, `.with_fallbacks()`
- **Cleaner composition** — Easy to combine chains, add steps
- **Debugging** — Better error messages, tracing
- **Streaming support** — Enable streaming responses naturally
- **Production-ready** — Handles async, batch operations

### Q2: How would you run multiple LLM calls in parallel?

**Answer:** Use `RunnableParallel`:

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=prompt_summary | llm | parser,
    questions=prompt_questions | llm | parser,
    flashcards=prompt_flashcards | llm | parser,
)

result = parallel.invoke("topic")
# All three LLM calls run in parallel (if using async)
# result["summary"], result["questions"], result["flashcards"]
```

**Or with a dict syntax (which is RunnableParallel under the hood):**
```python
chain = {
    "summary": prompt_summary | llm | parser,
    "questions": prompt_questions | llm | parser,
} | merger_prompt | llm | parser
```

### Q3: When would you use RunnableBranch vs RunnableLambda?

**Answer:**

- **RunnableBranch:** When you need **conditional routing** to different chains
  ```python
  # Route to different LLM prompts based on input type
  router = RunnableBranch(
      (lambda x: x["type"] == "quiz", quiz_chain),
      (lambda x: x["type"] == "summary", summary_chain),
      default_chain
  )
  ```

- **RunnableLambda:** When you need **custom Python logic** between steps
  ```python
  # Custom preprocessing or postprocessing
  clean = RunnableLambda(lambda x: x.strip().lower())
  chain = clean | prompt | llm | parser
  ```

**Key difference:** RunnableBranch chooses between multiple chains; RunnableLambda runs a function.

### Q4: How do you handle API failures in production?

**Answer:** Use chaining methods for resilience:

```python
chain = (
    prompt
    | ChatGoogleGenerativeAI(model="gemini-2.5-flash")  # Primary
    | parser
).with_fallbacks([
    ChatGoogleGenerativeAI(model="gemini-1.5-flash")  # Backup model
]).with_retry(
    stop=lambda attempt: attempt >= 2  # Retry max 2 times
)

# If gemini-2.5-flash fails → tries gemini-1.5-flash
# If that fails → retries once more
# If still fails → raises exception (you handle in try/except)
```

---

## Summary: The Cheat Sheet in One Chain

You already know the basics. Here's everything in one pattern:

```python
from langchain_core.runnables import (
    RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough
)
from langchain_core.output_parsers import StrOutputParser

# Your complete toolkit
study_app = (
    # 1. Validate input with RunnableLambda
    RunnableLambda(validate)
    # 2. Add computed fields with assign
    | RunnablePassthrough.assign(context=get_context)
    # 3. Route based on conditions with RunnableBranch
    | RunnableBranch(
        (lambda x: x["include_quiz"], quiz_prompt),
        summary_prompt
    )
    # 4. Call LLM
    | ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    # 5. Parse output
    | StrOutputParser()
).with_fallbacks([backup_chain]).with_retry(stop_after_attempt=2)

# Use it
result = study_app.invoke({...})
```

That's **LCEL Patterns in one example:**
- `RunnableLambda` = custom functions
- `assign()` = add computed fields
- `RunnableBranch` = conditional routing
- `.with_fallbacks()` = production resilience
- `.with_retry()` = error recovery

---

## Next Steps

1. **Refactor your retriever.py** — identify RunnableLambdas, RunnableBranches
2. **Add retry logic** — wrap your LLM calls with `.with_retry()`
3. **Parallelize if possible** — use RunnableParallel for independent operations
4. **Test composability** — pull out chains and reuse them elsewhere

You're now ready for production LangChain patterns. Happy chaining!
