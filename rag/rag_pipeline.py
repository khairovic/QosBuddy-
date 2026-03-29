"""
QoSBuddy M6 — Phase 2: RAG Pipeline
=====================================
DSO3.2 — NOC-Grade Question Answering over the QoSBuddy Knowledge Base

Architecture:
  User question
    → ChromaDB semantic retrieval (top-k, metadata filtered)
    → Context assembly (anomaly events + causal insights + domain context)
    → Ollama local LLM (llama3.2 / mistral)
    → Structured NOC-style response (alert + action + counterfactual)

Design decisions:
  • LCEL (LangChain Expression Language) — composable, streaming-ready.
  • Three retrieval strategies:
      "anomaly"  → filter by doc_type=ANOMALY_EVENT, severity metadata
      "causal"   → filter by doc_type=CAUSAL_INSIGHT, root_cause metadata
      "general"  → no filter (searches all doc families)
  • Structured output prompt forces the LLM to return:
      alert_summary | recommended_action | counterfactual_impact | confidence
  • Caching: ChromaDB queries are deterministic — wrap at Streamlit layer (st.cache_resource).
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger("qosbuddy.rag")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CHROMA_DIR      = Path(os.environ.get("QOSBUDDY_CHROMA_DIR", "./chroma_store"))
COLLECTION_NAME = "qosbuddy_knowledge"
EMBED_MODEL     = "all-MiniLM-L6-v2"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = "llama3.2"      # fallback: mistral, phi3, gemma2

TOP_K_RETRIEVAL = 6               # documents retrieved per query
MAX_CONTEXT_CHARS = 6000          # trim context if too long for LLM window


# ─────────────────────────────────────────────
# 1. Vector store connection
# ─────────────────────────────────────────────

def get_chroma_collection() -> chromadb.Collection:
    """Connect to the persistent ChromaDB collection built by build_knowledge_base.py."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION_NAME)


def get_embed_model() -> SentenceTransformer:
    """Load the same embedding model used during KB construction."""
    return SentenceTransformer(EMBED_MODEL)


# ─────────────────────────────────────────────
# 2. Retrieval
# ─────────────────────────────────────────────

def retrieve_context(
    query: str,
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    mode: str = "general",
    ue_id: Optional[int] = None,
    severity_filter: Optional[str] = None,
    root_cause_filter: Optional[str] = None,
    n_results: int = TOP_K_RETRIEVAL,
) -> List[Dict[str, Any]]:
    """
    Semantic retrieval with optional metadata filtering.

    mode options:
      "general"  — search all doc types
      "anomaly"  — restrict to ANOMALY_EVENT documents
      "causal"   — restrict to CAUSAL_INSIGHT documents
      "domain"   — restrict to DOMAIN_CONTEXT documents
    """
    # Build embedding for query
    query_emb = embed_model.encode(
        [query], normalize_embeddings=True
    ).tolist()

    # Build ChromaDB where-clause
    where: Dict[str, Any] = {}

    if mode == "anomaly":
        where["doc_type"] = {"$eq": "ANOMALY_EVENT"}
    elif mode == "causal":
        where["doc_type"] = {"$eq": "CAUSAL_INSIGHT"}
    elif mode == "domain":
        where["doc_type"] = {"$eq": "DOMAIN_CONTEXT"}

    # Narrow further by metadata if provided
    if ue_id is not None and mode in ("general", "anomaly"):
        where["ue_id"] = {"$eq": ue_id}

    if severity_filter and mode in ("general", "anomaly"):
        where["severity"] = {"$eq": severity_filter}

    if root_cause_filter and mode in ("general", "causal"):
        where["root_cause"] = {"$eq": root_cause_filter}

    # ChromaDB requires non-empty where dict or None
    query_kwargs = dict(
        query_embeddings=query_emb,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    # Flatten results into list of dicts
    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({
            "document": doc,
            "metadata": meta,
            "similarity": round(1 - dist, 4),  # cosine distance → similarity
        })

    return docs


def format_context_for_llm(retrieved_docs: List[Dict]) -> str:
    """
    Assemble retrieved documents into a context string for the LLM prompt.
    Ordered by similarity (highest first). Trimmed to MAX_CONTEXT_CHARS.
    """
    parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        sim = doc["similarity"]
        dt  = doc["metadata"].get("doc_type", "UNKNOWN")
        parts.append(f"[Source {i} | {dt} | similarity={sim:.3f}]\n{doc['document']}")

    context = "\n\n" + ("─" * 60) + "\n\n".join(parts)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[...context trimmed...]"

    return context


# ─────────────────────────────────────────────
# 3. LLM integration (Ollama / LangChain)
# ─────────────────────────────────────────────

NOC_SYSTEM_PROMPT = """You are QoSBuddy NOC Assistant, an expert AI system for 5G network operations.
You answer questions using ONLY the retrieved context provided. Your answers must be:
  • Technical and precise — use real KPI values from the context
  • Actionable — always recommend a concrete next step
  • Structured — follow the exact output format below

OUTPUT FORMAT (always use these exact headers):
**🚨 Alert Summary:**
[1-2 sentences describing what anomaly was detected, which UEs are affected, and severity level]

**✅ Recommended Action:**
[specific technical action from root cause analysis — reference the causal chain]

**📊 Counterfactual Impact:**
[quantify the expected improvement if the recommended action is applied — use % reduction from context]

**🎯 Confidence:** [HIGH / MEDIUM / LOW based on how much evidence is in the context]

If the context does not contain enough information, say so clearly. Do not hallucinate KPI values.
"""

NOC_HUMAN_TEMPLATE = """Retrieved network context:
{context}

NOC Question: {question}

Answer following the structured output format:"""


def build_langchain_rag_chain(
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
):
    """
    Build a LangChain LCEL RAG chain using Ollama as the local LLM.

    Returns a callable: chain.invoke({"question": ..., "mode": ..., "ue_id": ...})

    Falls back gracefully if langchain_ollama is not installed.
    """
    try:
        from langchain_ollama import OllamaLLM
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    except ImportError:
        log.warning("langchain_ollama not installed — using direct Ollama HTTP fallback")
        return None

    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,        # low temperature → deterministic, professional answers
        num_predict=512,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", NOC_SYSTEM_PROMPT),
        ("human", NOC_HUMAN_TEMPLATE),
    ])

    # Retrieval step as a Runnable
    def retrieve_step(inputs: Dict) -> Dict:
        question = inputs["question"]
        mode     = inputs.get("mode", "general")
        ue_id    = inputs.get("ue_id", None)
        severity = inputs.get("severity_filter", None)
        rc       = inputs.get("root_cause_filter", None)

        docs = retrieve_context(
            query=question,
            collection=collection,
            embed_model=embed_model,
            mode=mode,
            ue_id=ue_id,
            severity_filter=severity,
            root_cause_filter=rc,
        )
        return {
            "question": question,
            "context":  format_context_for_llm(docs),
            "sources":  docs,
        }

    chain = (
        RunnableLambda(retrieve_step)
        | RunnablePassthrough.assign(
            answer=prompt | llm | StrOutputParser()
        )
    )

    return chain


# ─────────────────────────────────────────────
# 4. Direct HTTP fallback (no LangChain needed)
# ─────────────────────────────────────────────

def ask_ollama_direct(
    question: str,
    context: str,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """
    Fallback: call Ollama REST API directly without LangChain.
    Useful if langchain_ollama is not installed or for testing.
    """
    import json
    import urllib.request

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": NOC_SYSTEM_PROMPT},
            {"role": "user",   "content": NOC_HUMAN_TEMPLATE.format(
                context=context, question=question
            )},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result["message"]["content"]
    except Exception as e:
        return f"[Ollama unavailable: {e}]\n\nContext retrieved:\n{context[:1000]}"


# ─────────────────────────────────────────────
# 5. Main RAG query function (unified interface)
# ─────────────────────────────────────────────

def query_rag(
    question: str,
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    chain=None,
    mode: str = "general",
    ue_id: Optional[int] = None,
    severity_filter: Optional[str] = None,
    root_cause_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified RAG query interface used by both the Streamlit dashboard and FastAPI.

    Returns:
      {
        "answer":     str,   # structured NOC response
        "sources":    list,  # retrieved documents with metadata
        "question":   str,
        "mode":       str,
      }
    """
    # Retrieve context
    docs = retrieve_context(
        query=question,
        collection=collection,
        embed_model=embed_model,
        mode=mode,
        ue_id=ue_id,
        severity_filter=severity_filter,
        root_cause_filter=root_cause_filter,
    )
    context = format_context_for_llm(docs)

    # Generate answer
    if chain is not None:
        try:
            result = chain.invoke({
                "question": question,
                "mode": mode,
                "ue_id": ue_id,
                "severity_filter": severity_filter,
                "root_cause_filter": root_cause_filter,
            })
            answer = result.get("answer", "")
        except Exception as e:
            log.warning(f"LangChain chain failed: {e} — falling back to direct Ollama")
            answer = ask_ollama_direct(question, context)
    else:
        answer = ask_ollama_direct(question, context)

    return {
        "answer":   answer,
        "sources":  docs,
        "question": question,
        "mode":     mode,
    }


# ─────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collection  = get_chroma_collection()
    embed_model = get_embed_model()
    chain       = build_langchain_rag_chain(collection, embed_model)

    test_questions = [
        "What is the main root cause of critical anomalies in the network?",
        "UE 1 has critical severity. What action should I take?",
        "How much can packet loss be reduced if congestion is resolved?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}\nQ: {q}")
        result = query_rag(q, collection, embed_model, chain)
        print(result["answer"])
        print(f"\nSources: {len(result['sources'])} documents retrieved")