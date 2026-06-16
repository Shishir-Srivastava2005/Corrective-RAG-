<div align="center">

<h1>🧠 Agentic RAG</h1>

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


