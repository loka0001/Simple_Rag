# src/agent.py
"""
Agent definition, state management, and the full answer generation pipeline.

The agent uses a ReAct (Reasoning + Acting) loop via LangChain's
`initialize_agent`. The LLM backend is selected via LLM_PROVIDER in .env:
    - "ollama"  → fully local (no internet required)
    - "groq"    → cloud inference (fast)
    - "openai"  → OpenAI-compatible

When ENABLE_REASONING=true, all answers are accompanied by a structured
chain-of-thought trace, confidence score, and plain-language explanation.
Arabic questions are automatically detected and handled in Arabic.
"""

import logging
from typing import Optional

from langchain.agents import AgentExecutor, AgentType, initialize_agent

from src.config import get_settings
from src.llm_factory import get_llm, get_reasoning_llm
from src.reasoning_chain import ReasonedResponse, generate_reasoned_response
from src.retriever import reformulate_query_with_llm, retrieve
from src.tools import get_tools
from src.arabic_utils import detect_language

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def build_agent() -> AgentExecutor:
    """
    Construct and return a ReAct-style AgentExecutor.

    Uses the LLM backend configured in LLM_PROVIDER (.env).
    Includes guardrails: max iterations and graceful parse-error handling.

    Returns:
        An AgentExecutor instance ready for invoke() calls.
    """
    cfg = get_settings()
    llm = get_llm()
    tools = get_tools()

    logger.info(
        "Building ReAct agent | provider=%s | max_iterations=%d | tools=%s",
        cfg.llm_provider,
        cfg.agent_max_iterations,
        [t.name for t in tools],
    )

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=cfg.agent_max_iterations,
        early_stopping_method="generate",
    )


# ---------------------------------------------------------------------------
# Run interface
# ---------------------------------------------------------------------------

def run_agent(
    query: str,
    reformulate: bool = False,
    with_reasoning: Optional[bool] = None,
    agent_executor: Optional[AgentExecutor] = None,
) -> str:
    """
    Execute the agentic RAG pipeline for a user query.

    Pipeline:
        1. Detect language (Arabic / English / mixed).
        2. Optionally reformulate query with LLM.
        3. Run ReAct agent loop (tool calls + reasoning).
        4. If ENABLE_REASONING (or with_reasoning=True), generate a structured
           chain-of-thought response with confidence + explanation.
        5. Return formatted answer string.

    Args:
        query:          The raw user question (Arabic or English).
        reformulate:    Rephrase the query before running the agent.
        with_reasoning: Override ENABLE_REASONING from config.
        agent_executor: Pre-built executor (useful for testing / caching).

    Returns:
        The agent's final answer as a formatted string.
    """
    cfg = get_settings()
    use_reasoning = with_reasoning if with_reasoning is not None else cfg.enable_reasoning
    language = detect_language(query)

    logger.info("Agent query | language=%s | reasoning=%s | query='%s'",
                language, use_reasoning, query[:80])

    if agent_executor is None:
        agent_executor = build_agent()

    if reformulate:
        llm = get_llm()
        query = reformulate_query_with_llm(query, llm)

    # --- Step 1: Run ReAct agent to get raw answer + retrieve context ---
    try:
        result = agent_executor.invoke({"input": query})
        raw_answer = result.get("output", "")
        if not raw_answer:
            raise RuntimeError("Agent returned an empty response.")
    except Exception as exc:
        logger.error("Agent execution failed: %s", exc)
        return f"I encountered an error while processing your question: {exc}"

    # --- Step 2: Optionally wrap with reasoning chain ---
    if not use_reasoning:
        return raw_answer

    # Retrieve context for grounding the reasoning explanation
    try:
        context_docs = retrieve(query, top_k=5, rerank=True, compress=False)
    except Exception as exc:
        logger.warning("Could not retrieve context for reasoning chain: %s", exc)
        context_docs = [raw_answer]  # Use the agent's own answer as context fallback

    reasoning_llm = get_reasoning_llm()
    reasoned: ReasonedResponse = generate_reasoned_response(
        question=query,
        context_chunks=context_docs,
        llm=reasoning_llm,
        language=language if language in ("ar", "en") else "en",
    )

    return reasoned.format_for_display()
