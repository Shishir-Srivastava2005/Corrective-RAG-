import os
import re
from typing import List, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain & LangGraph Imports
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_unstructured import UnstructuredLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Load environment variables (TAVILY_API_KEY, UNSTRUCTURED_API_KEY, etc.)
load_dotenv()

# ==========================================
# 1. GLOBAL SETUP & DEPENDENCIES
# ==========================================
tavily = TavilySearchResults(max_results=5)
llm = ChatOllama(model="gemma4:31b-cloud", temperature=0)
filter_llm = ChatOllama(model="phi4-mini")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

global_vector_store = None
global_retriever = None

UPPER_TH = 0.7
LOWER_TH = 0.3

# ==========================================
# 2. STATE & SCHEMAS
# ==========================================
class State(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]            
    kept_strips: List[str]       
    refined_context: str         
    web_docs: List[Document]
    web_query: str          
    answer: str

class DocEvalScore(BaseModel):
    score: float = Field(description="A score from 0.0 to 1.0. Must be a number. 1.0 means highly relevant, 0.0 means completely irrelevant.")
    reason: str = Field(description="A short, concise reason explaining why you gave the score.")

class KeepOrDrop(BaseModel):
    keep: bool

class WebQuery(BaseModel):
    query: str

# ==========================================
# 3. PROMPTS & CHAINS
# ==========================================
doc_eval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict retrieval evaluator for RAG.\n"
        "You will be given ONE retrieved chunk and a question.\n"
        "Return a score in [0.0, 1.0].\n"
        "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
        "- 0.0: chunk is irrelevant\n"
        "Be conservative with high scores.\n"
        "Also return a short reason.\n"
        "Output JSON only."
    ),
    ("human", "Question: {question}\n\nChunk:\n{chunk}"),
])
doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)

filter_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict relevance filter.\n"
        "Return keep=true only if the sentence directly helps answer the question.\n"
        "Use ONLY the sentence. Output JSON only."
    ),
    ("human", "Question: {question}\n\nSentence:\n{sentence}"),
])
filter_chain = filter_prompt | filter_llm.with_structured_output(KeepOrDrop)

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user question into a web search query composed of keywords.\n"
        "Rules:\n"
        "- Keep it short (6–14 words).\n"
        "- If the question implies recency (e.g., recent/latest/last week/last month), add a constraint like (last 30 days).\n"
        "- Do NOT answer the question.\n"
        "- Return JSON with a single key: query"
    ),
    ("human", "Question: {question}"),
])
rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY using the provided context.\n"
        "If the context is empty or insufficient, say: 'I don't know.'"
    ),
    ("human", "Question: {question}\n\nRefined context:\n{refined_context}"),
])

# ==========================================
# 4. LANGGRAPH NODES
# ==========================================
def retrieve_node(state: State) -> State:
    q = state["question"]
    if not global_retriever:
        raise ValueError("Global retriever is not initialized.")
    return {"docs": global_retriever.invoke(q)}

def eval_each_doc_node(state: State) -> State:
    q = state["question"]
    scores: List[float] = []
    reasons: List[str] = []
    good: List[Document] = []

    for d in state["docs"]:
        out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
        scores.append(out.score)
        reasons.append(out.reason)

        if out.score > LOWER_TH:
            good.append(d)

    if any(s > UPPER_TH for s in scores):
        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
        }

    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunks scored < {LOWER_TH}. No chunk was sufficient.",
        }

    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}. Mixed relevance signals.",
    }

def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def refine(state: State) -> State:
    q = state["question"]

    if state.get("verdict") == "CORRECT":
        docs_to_use = state["good_docs"]
    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state["web_docs"]
    else:  # AMBIGUOUS
        docs_to_use = state["good_docs"] + state.get("web_docs", [])

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    strips = decompose_to_sentences(context)

    kept: List[str] = []
    for s in strips:
        if filter_chain.invoke({"question": q, "sentence": s}).keep:
            kept.append(s)

    refined_context = "\n".join(kept).strip()

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }

def rewrite_query_node(state: State) -> State:
    out = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}

def web_search_node(state: State) -> State:
    q = state.get("web_query") or state["question"]  
    results = tavily.invoke({"query": q})

    web_docs = []
    for r in results or []:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

    return {"web_docs": web_docs}

def generate(state: State) -> State:
    out = (answer_prompt | llm).invoke(
        {"question": state["question"], "refined_context": state["refined_context"]}
    )
    return {"answer": out.content}

def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"
    else:
        return "rewrite_query"

# ==========================================
# 5. GRAPH COMPILATION
# ==========================================
g = StateGraph(State)

g.add_node("retrieve", retrieve_node)
g.add_node("eval_each_doc", eval_each_doc_node)
g.add_node("rewrite_query", rewrite_query_node)
g.add_node("web_search", web_search_node)
g.add_node("refine", refine)
g.add_node("generate", generate)

g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_doc")

g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {
        "refine": "refine",
        "rewrite_query": "rewrite_query",
    },
)

# Non-correct/ambiguous path
g.add_edge("rewrite_query", "web_search")
g.add_edge("web_search", "refine")

# Correct path 
g.add_edge("refine", "generate")
g.add_edge("generate", END)

rag_app = g.compile()


# ==========================================
# 6. EXTERNAL API FUNCTIONS
# ==========================================
def process_document(file_path: str) -> int:
    """Handles Unstructured chunking and Chroma DB embedding."""
    global global_vector_store, global_retriever
    
    unstructured_key = os.getenv("UNSTRUCTURED_API_KEY")
    unstructured_url = os.getenv("UNSTRUCTURED_URL")
    
    loader = UnstructuredLoader(
        file_path=file_path,
        partition_via_api=True,
        api_key=unstructured_key,
        url=unstructured_url,
        strategy="hi_res",           
        extract_images_in_pdf=True, 
        infer_table_structure=True,  
        chunking_strategy="basic", 
        max_characters=1000,
        combine_text_under_n_chars=500
    )
    
    docs = loader.load()
    
    # Initialize Chroma and reset to handle fresh document uploads dynamically
    global_vector_store = Chroma(
        collection_name="rag_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    global_vector_store.reset_collection()
    
    BATCH_SIZE = 100
    total_docs = len(docs)
    
    for i in range(0, total_docs, BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        global_vector_store.add_documents(documents=batch)
        
    global_retriever = global_vector_store.as_retriever(
        search_type='similarity', 
        search_kwargs={'k': 4}
    )
    
    return total_docs

def chat_with_agent(question: str) -> dict:
    """Invokes the LangGraph state machine."""
    if not global_retriever:
        raise ValueError("Database is empty. Please upload a PDF first.")
        
    res = rag_app.invoke(
        {
            "question": question,
            "docs": [],
            "good_docs": [],
            "verdict": "",
            "reason": "",
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
            "web_query": "",
            "web_docs": [],
            "answer": "",
        }
    )
    
    return {
        "answer": res.get("answer"),
        "verdict": res.get("verdict"),
        "reason": res.get("reason"),
        "web_query": res.get("web_query")
    }