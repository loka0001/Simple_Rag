"""
Centralized configuration module.

Loads all settings from environment variables (via a .env file) using
python-dotenv. No secrets or magic numbers are hardcoded here.

Supported LLM providers:
    - groq   : Cloud-hosted (fast inference, Mixtral / Llama)
    - ollama : Local models (fully offline, privacy-preserving)
    - openai : OpenAI-compatible endpoints
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load .env file from the project root
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


class Settings(BaseSettings):
    """Application settings validated by Pydantic."""

    # ------------------------------------------------------------------ #
    # LLM Provider selection
    # ------------------------------------------------------------------ #
    # Set LLM_PROVIDER to "groq", "ollama", or "openai"
    llm_provider: Literal["groq", "ollama", "openai"] = Field("groq", alias="LLM_PROVIDER")

    # --- Groq ---
    groq_api_key: str = Field("", alias="GROQ_API_KEY")
    groq_model: str = Field("mixtral-8x7b-32768", alias="GROQ_MODEL")

    # --- Ollama (local) ---
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3", alias="OLLAMA_MODEL")
    # Reasoning-capable model for the explain/reasoning chain
    ollama_reasoning_model: str = Field("deepseek-r1:8b", alias="OLLAMA_REASONING_MODEL")

    # --- OpenAI ---
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", alias="OPENAI_MODEL")

    # Shared LLM params
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(2048, alias="LLM_MAX_TOKENS")

    # ------------------------------------------------------------------ #
    # Reasoning / Explanation feature
    # ------------------------------------------------------------------ #
    # When True, responses include a step-by-step reasoning trace
    enable_reasoning: bool = Field(True, alias="ENABLE_REASONING")
    # Language to use for reasoning output: "auto", "ar", "en"
    reasoning_language: str = Field("auto", alias="REASONING_LANGUAGE")

    # ------------------------------------------------------------------ #
    # HuggingFace
    # ------------------------------------------------------------------ #
    hugging_face_api_key: str = Field("", alias="HUGGING_FACE_API_KEY")

    # ------------------------------------------------------------------ #
    # Embedding & Reranking
    # ------------------------------------------------------------------ #
    # Use a multilingual model by default to handle Arabic + English
    embedding_model: str = Field(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL",
    )
    rerank_model: str = Field("bert-base-multilingual-cased", alias="RERANK_MODEL")

    # ------------------------------------------------------------------ #
    # Arabic / multilingual settings
    # ------------------------------------------------------------------ #
    # Encoding to attempt when reading Arabic text files
    arabic_encodings: list[str] = Field(
        default=["utf-8", "windows-1256", "iso-8859-6", "cp1256"],
        alias="ARABIC_ENCODINGS",
    )
    # Normalize Arabic text (remove tashkeel, normalize alef/yaa/ta marbuta)
    normalize_arabic: bool = Field(True, alias="NORMALIZE_ARABIC")

    # ------------------------------------------------------------------ #
    # ChromaDB
    # ------------------------------------------------------------------ #
    chroma_persist_dir: str = Field("./data/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("documents", alias="CHROMA_COLLECTION_NAME")

    # ------------------------------------------------------------------ #
    # Elasticsearch
    # ------------------------------------------------------------------ #
    es_host: str = Field("", alias="ES_HOST")
    es_username: str = Field("elastic", alias="ES_USERNAME")
    es_password: str = Field("", alias="ES_PASSWORD")
    es_index_name: str = Field("documents", alias="ES_INDEX_NAME")

    # ------------------------------------------------------------------ #
    # Neo4j
    # ------------------------------------------------------------------ #
    neo4j_uri: str = Field("", alias="NEO4J_URI")
    neo4j_username: str = Field("neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field("", alias="NEO4J_PASSWORD")

    # ------------------------------------------------------------------ #
    # Search tools
    # ------------------------------------------------------------------ #
    serper_api_key: str = Field("", alias="SERPER_API_KEY")

    # ------------------------------------------------------------------ #
    # LangSmith (optional tracing)
    # ------------------------------------------------------------------ #
    langchain_tracing_v2: str = Field("false", alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field("", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field("agentic-rag", alias="LANGCHAIN_PROJECT")

    # ------------------------------------------------------------------ #
    # Agent guardrails
    # ------------------------------------------------------------------ #
    agent_max_iterations: int = Field(5, alias="AGENT_MAX_ITERATIONS")

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")

    class Config:
        env_file = ".env"
        populate_by_name = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()
