"""
Retrieval layer for the Agentic RAG system.

Implements:
- Vector retrieval via ChromaDB (dense, semantic search).
- Keyword retrieval via Elasticsearch (BM25 / textual search).
- Fusion retrieval combining both methods (Reciprocal Rank Fusion).
- Cross-encoder reranking for result refinement.
- Context compression (extractive summarization).
- Query routing to select the appropriate retrieval strategy.
- Query expansion / transformation for improved recall.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    pipeline,
)

from src.config import get_settings
from src.ingest import get_chroma_collection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query Transformation
# ---------------------------------------------------------------------------

def expand_query(query: str, synonyms: Optional[List[str]] = None) -> str:
    """
    Expand a query with optional synonyms or related terms to improve recall.

    Args:
        query:    The original user query.
        synonyms: Optional list of related terms to append.

    Returns:
        The expanded query string.
    """
    extra = " OR ".join(synonyms) if synonyms else ""
    expanded = f"{query} {extra}".strip()
    logger.debug("Expanded query: '%s'", expanded)
    return expanded


def reformulate_query_with_llm(query: str, llm) -> str:
    """
    Use an LLM to reformulate the query for better retrieval (HyDE / step-back).

    Args:
        query: Original user query.
        llm:   A LangChain-compatible chat model instance.

    Returns:
        A reformulated query string.
    """
    try:
        prompt = (
            "Rewrite the following question to make it more specific and search-friendly. "
            "Return ONLY the rewritten question, nothing else.\n\n"
            f"Original: {query}\nRewritten:"
        )
        response = llm.invoke(prompt)
        reformulated = response.content.strip()
        logger.info("Reformulated query: '%s'", reformulated)
        return reformulated
    except Exception as exc:
        logger.warning("Query reformulation failed (%s); using original.", exc)
        return query


# ---------------------------------------------------------------------------
# Query Routing
# ---------------------------------------------------------------------------

TEXTUAL_KEYWORDS = {"who", "what year", "when", "where", "define", "list", "how many"}


def route_query(query: str) -> str:
    """
    Decide the retrieval strategy based on query characteristics.

    Uses a simple keyword heuristic: fact-seeking queries (who/what/when) are
    routed to textual (BM25) retrieval; all others use vector (semantic) search.
    Fusion retrieval is used when neither heuristic fires clearly.

    Args:
        query: The user's question.

    Returns:
        One of "textual", "vector", or "fusion".
    """
    q_lower = query.lower()
    if any(kw in q_lower for kw in TEXTUAL_KEYWORDS):
        logger.debug("Routing query to 'textual' retriever.")
        return "textual"
    logger.debug("Routing query to 'vector' retriever.")
    return "vector"


# ---------------------------------------------------------------------------
# Vector Retrieval (ChromaDB)
# ---------------------------------------------------------------------------

def vector_retrieve(
    query: str,
    top_k: int = 5,
    collection_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> List[str]:
    """
    Retrieve the top_k most semantically similar documents from ChromaDB.

    Args:
        query:            The search query.
        top_k:            Number of results to retrieve.
        collection_name:  Override the default ChromaDB collection name.
        embedding_model:  Override the default embedding model.

    Returns:
        A list of document text strings, ordered by similarity.
    """
    cfg = get_settings()
    try:
        model = SentenceTransformer(embedding_model or cfg.embedding_model)
        collection = get_chroma_collection(collection_name)
        query_embedding = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count() or top_k),
        )
        docs = results.get("documents", [[]])[0]
        logger.info("Vector retrieval returned %d documents.", len(docs))
        return docs
    except Exception as exc:
        logger.error("Vector retrieval failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Textual Retrieval (Elasticsearch / BM25)
# ---------------------------------------------------------------------------

def textual_retrieve(
    query: str,
    top_k: int = 5,
    index_name: Optional[str] = None,
) -> List[str]:
    """
    Retrieve documents from Elasticsearch using BM25 keyword matching.

    Args:
        query:      The search query.
        top_k:      Number of results to retrieve.
        index_name: Override the default Elasticsearch index name.

    Returns:
        A list of document text strings, ordered by BM25 score.
    """
    from elasticsearch import Elasticsearch

    cfg = get_settings()
    if not cfg.es_host:
        logger.warning("ES_HOST not configured; textual retrieval unavailable.")
        return []

    try:
        es = Elasticsearch(
            hosts=[cfg.es_host],
            http_auth=(cfg.es_username, cfg.es_password),
        )
        idx = index_name or cfg.es_index_name
        response = es.search(
            index=idx,
            body={"size": top_k, "query": {"match": {"content": query}}},
        )
        docs = [hit["_source"]["content"] for hit in response["hits"]["hits"]]
        logger.info("Textual retrieval returned %d documents.", len(docs))
        return docs
    except Exception as exc:
        logger.error("Textual retrieval failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Fusion Retrieval (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    ranked_lists: List[List[str]], k: int = 60
) -> List[str]:
    """
    Merge multiple ranked document lists using Reciprocal Rank Fusion (RRF).

    Args:
        ranked_lists: Each inner list is a ranked list of document strings.
        k:            RRF damping constant (default 60, per the original paper).

    Returns:
        A single merged, re-ranked list of unique document strings.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def fusion_retrieve(query: str, top_k: int = 5) -> List[str]:
    """
    Combine vector and textual retrieval using Reciprocal Rank Fusion.

    Args:
        query:  The search query.
        top_k:  Number of results to return from the merged list.

    Returns:
        A fused, ranked list of document strings.
    """
    vector_docs = vector_retrieve(query, top_k=top_k)
    textual_docs = textual_retrieve(query, top_k=top_k)
    fused = _reciprocal_rank_fusion([vector_docs, textual_docs])
    logger.info("Fusion retrieval produced %d unique documents.", len(fused))
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Cross-Encoder Reranking
# ---------------------------------------------------------------------------

_rerank_tokenizer = None
_rerank_model = None


def _load_rerank_model():
    """Lazy-load the cross-encoder reranking model."""
    global _rerank_tokenizer, _rerank_model
    if _rerank_tokenizer is None:
        cfg = get_settings()
        logger.info("Loading rerank model '%s'...", cfg.rerank_model)
        _rerank_tokenizer = AutoTokenizer.from_pretrained(cfg.rerank_model)
        _rerank_model = BertForSequenceClassification.from_pretrained(cfg.rerank_model)
        _rerank_model.eval()


def rerank_documents(query: str, documents: List[str]) -> List[str]:
    """
    Rerank a list of documents by relevance to the query using a cross-encoder.

    Args:
        query:     The user query used as relevance anchor.
        documents: Candidate documents to rerank.

    Returns:
        Documents sorted from most to least relevant.
    """
    if not documents:
        return []

    try:
        _load_rerank_model()
        scores: List[Tuple[str, float]] = []
        for doc in documents:
            inputs = _rerank_tokenizer.encode_plus(
                query, doc, return_tensors="pt", truncation=True, padding=True, max_length=512
            )
            with torch.no_grad():
                logits = _rerank_model(**inputs).logits
            prob = F.softmax(logits, dim=1)[:, 1].item()
            scores.append((doc, prob))

        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        logger.info("Reranking complete; top score=%.4f.", ranked[0][1] if ranked else 0.0)
        return [doc for doc, _ in ranked]
    except Exception as exc:
        logger.error("Reranking failed: %s", exc)
        return documents  # Fallback: return original order


# ---------------------------------------------------------------------------
# Context Compression
# ---------------------------------------------------------------------------

_summarizer = None


def _load_summarizer():
    """Lazy-load the summarization pipeline."""
    global _summarizer
    if _summarizer is None:
        logger.info("Loading summarization pipeline...")
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def compress_context(documents: List[str], max_words: int = 100) -> List[str]:
    """
    Summarize each document to reduce context length fed to the LLM.

    Args:
        documents: Retrieved documents to compress.
        max_words: Approximate maximum length in words for each summary.

    Returns:
        A list of summarized document strings.
    """
    if not documents:
        return []

    try:
        _load_summarizer()
        compressed: List[str] = []
        for doc in documents:
            word_count = len(doc.split())
            if word_count <= max_words:
                compressed.append(doc)
                continue
            max_len = min(max_words, word_count)
            min_len = max(5, max_len // 4)
            summary = _summarizer(
                doc, max_length=max_len, min_length=min_len, do_sample=False
            )[0]["summary_text"]
            compressed.append(summary)
        return compressed
    except Exception as exc:
        logger.error("Context compression failed: %s", exc)
        return documents


# ---------------------------------------------------------------------------
# Unified Retrieval Interface
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = 5,
    rerank: bool = True,
    compress: bool = True,
    strategy: Optional[str] = None,
) -> List[str]:
    """
    High-level retrieval function used by the agent.

    Performs query routing (or uses a specified strategy), retrieves documents,
    optionally reranks and compresses them.

    Args:
        query:    The user's search query.
        top_k:    Number of documents to return.
        rerank:   Apply cross-encoder reranking when True.
        compress: Apply context compression when True.
        strategy: Force a specific strategy ("vector", "textual", "fusion").

    Returns:
        A processed list of relevant document strings.
    """
    method = strategy or route_query(query)

    if method == "textual":
        docs = textual_retrieve(query, top_k=top_k)
    elif method == "fusion":
        docs = fusion_retrieve(query, top_k=top_k)
    else:  # default: vector
        docs = vector_retrieve(query, top_k=top_k)

    if not docs:
        logger.warning("No documents retrieved for query: '%s'", query)
        return []

    if rerank:
        docs = rerank_documents(query, docs)

    if compress:
        docs = compress_context(docs)

    return docs[:top_k]
