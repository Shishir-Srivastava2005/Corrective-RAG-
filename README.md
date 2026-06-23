
<h1>Agentic RAG</h1>

<p><em>A Corrective RAG system that evaluates its own retrieval — and falls back to the web when local knowledge isn't enough.</em></p>
<p><img width="592" height="412" alt="image" src="https://github.com/user-attachments/assets/0a6148ba-7903-4eca-a4de-6181dda0a831" /></p>

<h2>📌 Overview</h2>

<p>
Most RAG systems retrieve chunks and blindly pass them to an LLM. This system doesn't.
</p>

<p>
Before generating any answer, an LLM evaluator scores every retrieved chunk on a <b>0.0 → 1.0 relevance scale</b>. Based on those scores, the LangGraph pipeline routes the request down one of three paths — using local documents, falling back to live web search, or intelligently combining both. A sentence-level filter then strips noise from the final context before generation.
</p>

<p>
This is an implementation of the <b>Corrective RAG (CRAG)</b> pattern, served through a FastAPI backend and fully orchestrated with LangGraph state machines.
</p>

---
<h2>✨ What Makes This Different from Vanilla RAG</h2>

<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Vanilla RAG</th>
      <th>This System</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Chunk evaluation</td>
      <td>❌ None</td>
      <td>✅ LLM scores each chunk 0.0–1.0</td>
    </tr>
    <tr>
      <td>Routing logic</td>
      <td>❌ Always uses local docs</td>
      <td>✅ <code>CORRECT</code> / <code>AMBIGUOUS</code> / <code>INCORRECT</code> verdicts</td>
    </tr>
    <tr>
      <td>Web search fallback</td>
      <td>❌ None</td>
      <td>✅ Automatic via Tavily when retrieval fails</td>
    </tr>
    <tr>
      <td>Query rewriting</td>
      <td>❌ None</td>
      <td>✅ LLM rewrites question into optimized search keywords</td>
    </tr>
    <tr>
      <td>Context refinement</td>
      <td>❌ Full chunk passed as-is</td>
      <td>✅ Sentence-level LLM filter strips irrelevant sentences</td>
    </tr>
  </tbody>
</table>

---

<h2>🏗️ Architecture</h2>
<img width="198" height="729" alt="image" src="https://github.com/user-attachments/assets/0696b8bb-635b-4737-8be8-9180367c7b1c" />
<p>The full pipeline is a LangGraph <code>StateGraph</code> with conditional routing after the evaluation node:</p>

```
User Question
      │
      ▼
 ┌──────────┐
 │ retrieve │  ←  ChromaDB similarity search (k=4)
 └────┬─────┘
      │
      ▼
 ┌───────────────┐
 │ eval_each_doc │  ←  LLM scores each chunk [0.0 → 1.0]
 └───────┬───────┘
         │
  ┌──────▼──────┐
  │    route    │
  └──┬──────┬───┘
     │      │
  ≥ 0.7   < 0.3 or mixed
(CORRECT) (INCORRECT / AMBIGUOUS)
     │            │
     ▼            ▼
  ┌──────┐  ┌──────────────┐
  │refine│  │ rewrite_query│  ←  LLM rewrites for web search
  └──┬───┘  └──────┬───────┘
     │             │
     │      ┌──────▼──────┐
     │      │  web_search │  ←  Tavily (top 5 results)
     │      └──────┬──────┘
     │             │
     │      ┌──────▼──────┐
     └──────►    refine   │  ←  Sentence-level LLM filter (phi4-mini)
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │   generate  │  ←  Final synthesis (gemma4:31b)
            └──────┬──────┘
                   │
                 Answer
```
## 🔍 Observability & Tracing

Every graph execution is fully traced via **LangSmith**. Each LangGraph node — `retrieve`, `eval_each_doc`, `rewrite_query`, `web_search`, `refine`, `generate` — appears as a named span with its inputs, outputs, and latency.

This makes it possible to:
- See exactly which verdict path was taken (CORRECT / AMBIGUOUS / INCORRECT) for any query
- Inspect what score each chunk received in `eval_each_doc`
- Compare runs across different documents or threshold configs

<br/>

<h3>Verdict Logic</h3>

<table>
  <thead>
    <tr>
      <th>Verdict</th>
      <th>Condition</th>
      <th>Path Taken</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>CORRECT</code></td>
      <td>At least 1 chunk scored <code>&gt; 0.7</code></td>
      <td>Refine local docs → Generate</td>
    </tr>
    <tr>
      <td><code>INCORRECT</code></td>
      <td>All chunks scored <code>&lt; 0.3</code></td>
      <td>Rewrite query → Web search → Refine → Generate</td>
    </tr>
    <tr>
      <td><code>AMBIGUOUS</code></td>
      <td>Mixed scores in the 0.3–0.7 range</td>
      <td>Rewrite query → Web search → Merge with local docs → Refine → Generate</td>
    </tr>
  </tbody>
</table>

---
<h2>🔬 How Context Refinement Works</h2>

<p>
The <code>refine</code> node runs after retrieval (and after web search if triggered). Its job is to strip noise from the context before generation. It works in three steps:
</p>

<table>
  <thead>
    <tr><th>Step</th><th>What Happens</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><b>1. Decompose</b></td>
      <td>The full context is split into individual sentences using a regex boundary splitter <code>(?&lt;=[.!?])\s+</code></td>
    </tr>
    <tr>
      <td><b>2. LLM Filter</b></td>
      <td>Each sentence is independently judged by <code>phi4-mini</code> via structured output — returns <code>keep: true/false</code></td>
    </tr>
    <tr>
      <td><b>3. Reconstruct</b></td>
      <td>Only kept sentences are joined and passed to the generator as <code>refined_context</code></td>
    </tr>
  </tbody>
</table>

<p>
This is especially important on the <code>AMBIGUOUS</code> path, where local docs and web results are merged — without refinement, the generator would be flooded with conflicting, off-topic noise.
</p>

---

<h2>⚙️ Configuration</h2>

<p>Two thresholds in <code>rag_engine.py</code> control the routing behaviour:</p>

```python
UPPER_TH = 0.7   # At least one chunk above this → CORRECT (local docs are good enough)
LOWER_TH = 0.3   # All chunks below this → INCORRECT (fall back entirely to web)
```

<table>
  <thead>
    <tr><th>If you want to…</th><th>Then…</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>Be stricter about local retrieval quality</td>
      <td>Increase <code>UPPER_TH</code> (e.g. <code>0.8</code>)</td>
    </tr>
    <tr>
      <td>Tolerate weaker chunks before triggering web search</td>
      <td>Lower <code>LOWER_TH</code> (e.g. <code>0.2</code>)</td>
    </tr>
    <tr>
      <td>Force more web search usage</td>
      <td>Increase both thresholds so the AMBIGUOUS band widens</td>
    </tr>
  </tbody>
</table>

---

<h2>🛠️ Tech Stack</h2>

<table>
  <thead>
    <tr>
      <th>Layer</th>
      <th>Technology</th>
      <th>Role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Orchestration</td>
      <td><a href="https://github.com/langchain-ai/langgraph">LangGraph</a></td>
      <td>State machine, conditional routing, node execution</td>
    </tr>
    <tr>
      <td>API</td>
      <td><a href="https://fastapi.tiangolo.com/">FastAPI</a></td>
      <td>REST endpoints for upload and chat</td>
    </tr>
    <tr>
      <td>LLM — Main</td>
      <td><a href="https://ollama.com/">Ollama</a> · <code>gemma4:31b-cloud</code></td>
      <td>Chunk evaluation, query rewriting, answer generation</td>
    </tr>
    <tr>
      <td>LLM — Filter</td>
      <td><a href="https://ollama.com/">Ollama</a> · <code>phi4-mini</code></td>
      <td>Sentence-level keep/drop filtering (fast, lightweight)</td>
    </tr>
    <tr>
      <td>Embeddings</td>
      <td><a href="https://ollama.com/">Ollama</a> · <code>nomic-embed-text</code></td>
      <td>Document and query embedding</td>
    </tr>
    <tr>
      <td>Vector Store</td>
      <td><a href="https://www.trychroma.com/">ChromaDB</a></td>
      <td>Persistent local vector storage with similarity search</td>
    </tr>
    <tr>
      <td>Document Parsing</td>
      <td><a href="https://unstructured.io/">Unstructured.io</a></td>
      <td>Hi-res PDF parsing, table extraction, image extraction</td>
    </tr>
    <tr>
      <td>Web Search</td>
      <td><a href="https://tavily.com/">Tavily</a></td>
      <td>Fallback web retrieval (top 5 results)</td>
    </tr>
  </tbody>
</table>

