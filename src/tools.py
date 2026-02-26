# src/tools.py
"""
Custom LangChain tools available to the Agentic RAG agent.

Tools:
- RAGRetrievalTool  : Retrieves context from the vector / hybrid store.
- WebSearchTool     : DuckDuckGo-based fallback web search.
- WikipediaTool     : Wikipedia lookup for factual questions.
- CalculatorTool    : Safe numeric expression evaluator.
- GraphQueryTool    : Query the Neo4j knowledge graph via natural language.
"""

import logging
import math
from typing import Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import get_settings
from src.retriever import retrieve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input Schemas (Pydantic)
# ---------------------------------------------------------------------------

class QueryInput(BaseModel):
    """Input schema for query-based tools."""
    query: str = Field(..., description="The search or question query.")


class ExpressionInput(BaseModel):
    """Input schema for the calculator tool."""
    expression: str = Field(..., description="A safe mathematical expression, e.g. '2 * (3 + 4)'.")


# ---------------------------------------------------------------------------
# RAG Retrieval Tool
# ---------------------------------------------------------------------------

class RAGRetrievalTool(BaseTool):
    """
    Retrieve relevant documents from the local knowledge base.

    Uses semantic + keyword fusion retrieval with cross-encoder reranking.
    Falls back gracefully if the vector store is unavailable.
    """

    name: str = "rag_retrieval"
    description: str = (
        "Use this tool to look up relevant information from the internal knowledge base. "
        "Input should be a specific question or search query."
    )
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, query: str) -> str:
        try:
            docs = retrieve(query, top_k=5, rerank=True, compress=True)
            if not docs:
                return "No relevant documents found in the knowledge base."
            return "\n\n---\n\n".join(docs)
        except Exception as exc:
            logger.error("RAGRetrievalTool error: %s", exc)
            return f"Knowledge base retrieval failed: {exc}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


# ---------------------------------------------------------------------------
# Web Search Tool
# ---------------------------------------------------------------------------

class WebSearchTool(BaseTool):
    """
    Search the web using DuckDuckGo for recent or external information.

    Used as a fallback when the knowledge base has no relevant results.
    """

    name: str = "web_search"
    description: str = (
        "Search the internet using DuckDuckGo for up-to-date information. "
        "Use when the knowledge base doesn't have an answer."
    )
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, query: str) -> str:
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            result = search.run(query)
            logger.info("Web search completed for query: '%s'", query)
            return result
        except Exception as exc:
            logger.error("WebSearchTool error: %s", exc)
            return f"Web search failed: {exc}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


# ---------------------------------------------------------------------------
# Wikipedia Tool
# ---------------------------------------------------------------------------

class WikipediaTool(BaseTool):
    """
    Look up factual information on Wikipedia.

    Best for biographical, historical, or definitional queries.
    """

    name: str = "wikipedia_search"
    description: str = (
        "Search Wikipedia for factual, biographical, or encyclopedic information. "
        "Useful for historical events, definitions, and well-known entities."
    )
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, query: str) -> str:
        try:
            from langchain_community.tools import WikipediaQueryRun
            from langchain_community.utilities import WikipediaAPIWrapper
            wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
            result = wiki.run(query)
            logger.info("Wikipedia search completed for query: '%s'", query)
            return result
        except Exception as exc:
            logger.error("WikipediaTool error: %s", exc)
            return f"Wikipedia search failed: {exc}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


# ---------------------------------------------------------------------------
# Calculator Tool
# ---------------------------------------------------------------------------

class CalculatorTool(BaseTool):
    """
    Evaluate safe mathematical expressions.

    Supports standard arithmetic, exponentiation, and common math functions.
    Rejects any expression containing non-numeric characters to prevent injection.
    """

    name: str = "calculator"
    description: str = (
        "Evaluate mathematical expressions. "
        "Input must be a valid numeric expression like '2 ** 10' or 'sqrt(144)'."
    )
    args_schema: Type[BaseModel] = ExpressionInput
    _SAFE_NAMES: dict = {
        "abs": abs, "round": round,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e,
    }

    def _run(self, expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": {}}, self._SAFE_NAMES)  # noqa: S307
            return str(result)
        except Exception as exc:
            logger.warning("CalculatorTool failed for '%s': %s", expression, exc)
            return f"Could not evaluate expression: {exc}"

    async def _arun(self, expression: str) -> str:
        return self._run(expression)


# ---------------------------------------------------------------------------
# Knowledge Graph Query Tool
# ---------------------------------------------------------------------------

class GraphQueryTool(BaseTool):
    """
    Query the Neo4j knowledge graph using natural language.

    Converts the user's question into a Cypher query via an LLM, executes it
    against Neo4j, and returns the results as a formatted string.
    Gracefully handles missing Neo4j configuration.
    """

    name: str = "knowledge_graph_query"
    description: str = (
        "Query the Neo4j knowledge graph for structured relationship data. "
        "Use for questions about entities and their connections (e.g., 'Who works with X?')."
    )
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, query: str) -> str:
        cfg = get_settings()
        if not cfg.neo4j_uri or not cfg.neo4j_password:
            return "Knowledge graph is not configured (NEO4J_URI / NEO4J_PASSWORD missing)."
        try:
            from langchain_community.graphs import Neo4jGraph
            from langchain.chains import GraphCypherQAChain
            from src.llm_factory import get_llm

            graph = Neo4jGraph(
                url=cfg.neo4j_uri,
                username=cfg.neo4j_username,
                password=cfg.neo4j_password,
            )
            llm = get_llm()
            chain = GraphCypherQAChain.from_llm(llm, graph=graph, verbose=False)
            result = chain.invoke({"query": query})
            return result.get("result", "No result from knowledge graph.")
        except Exception as exc:
            logger.error("GraphQueryTool error: %s", exc)
            return f"Knowledge graph query failed: {exc}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

def get_tools() -> list:
    """
    Return all tools available to the agent.

    Returns:
        A list of instantiated LangChain tool objects.
    """
    return [
        RAGRetrievalTool(),
        WebSearchTool(),
        WikipediaTool(),
        CalculatorTool(),
        GraphQueryTool(),
    ]
