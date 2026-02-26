"""
LLM Factory — provider-agnostic language model instantiation.

Supported backends:
    - Groq  : Fast cloud inference (Mixtral, Llama-3, Gemma)
    - Ollama: Fully local inference (llama3, mistral, deepseek-r1, phi3, …)
    - OpenAI: OpenAI-compatible endpoints (GPT-4o, GPT-3.5, etc.)

Usage:
    from src.llm_factory import get_llm, get_reasoning_llm

    llm = get_llm()                   # Primary LLM for answers
    reasoning_llm = get_reasoning_llm()  # Heavier model for chain-of-thought
"""

import logging
import os
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)


def get_llm(provider: str | None = None) -> Any:
    """
    Instantiate the primary LLM based on the configured (or overridden) provider.

    Args:
        provider: Force a specific backend ("groq", "ollama", "openai").
                  Defaults to the LLM_PROVIDER env setting.

    Returns:
        A LangChain-compatible chat model instance.

    Raises:
        ValueError: If an unsupported provider is specified.
        RuntimeError: If required credentials are missing.
    """
    cfg = get_settings()
    backend = (provider or cfg.llm_provider).lower()

    logger.info("Initializing LLM with provider='%s'", backend)

    if backend == "ollama":
        return _make_ollama_llm(model=cfg.ollama_model)

    if backend == "groq":
        return _make_groq_llm(model=cfg.groq_model)

    if backend == "openai":
        return _make_openai_llm(model=cfg.openai_model)

    raise ValueError(
        f"Unsupported LLM provider: '{backend}'. "
        "Valid options: 'groq', 'ollama', 'openai'."
    )


def get_reasoning_llm(provider: str | None = None) -> Any:
    """
    Instantiate a reasoning-optimized LLM for chain-of-thought and explanation tasks.

    For Ollama this defaults to a dedicated reasoning model (e.g. deepseek-r1:8b).
    For cloud providers it reuses the primary model with a higher temperature.

    Args:
        provider: Force a specific backend.

    Returns:
        A LangChain-compatible chat model instance tuned for reasoning.
    """
    cfg = get_settings()
    backend = (provider or cfg.llm_provider).lower()

    logger.info("Initializing reasoning LLM with provider='%s'", backend)

    if backend == "ollama":
        return _make_ollama_llm(model=cfg.ollama_reasoning_model, temperature=0.1)

    if backend == "groq":
        return _make_groq_llm(model=cfg.groq_model, temperature=0.1)

    if backend == "openai":
        return _make_openai_llm(model=cfg.openai_model, temperature=0.1)

    raise ValueError(f"Unsupported reasoning LLM provider: '{backend}'.")


# ---------------------------------------------------------------------------
# Private constructors
# ---------------------------------------------------------------------------

def _make_ollama_llm(model: str, temperature: float | None = None) -> Any:
    """
    Build a ChatOllama instance for local inference.

    Ollama must be running at OLLAMA_BASE_URL (default: http://localhost:11434).
    Start it with: ``ollama serve`` and pull a model with: ``ollama pull llama3``

    Args:
        model:       The Ollama model tag (e.g. "llama3", "mistral", "deepseek-r1:8b").
        temperature: Override the config temperature.

    Returns:
        A ChatOllama instance.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama  # type: ignore

    cfg = get_settings()
    llm = ChatOllama(
        model=model,
        base_url=cfg.ollama_base_url,
        temperature=temperature if temperature is not None else cfg.llm_temperature,
        num_predict=cfg.llm_max_tokens,
    )
    logger.info("Ollama LLM ready: model='%s' @ %s", model, cfg.ollama_base_url)
    return llm


def _make_groq_llm(model: str, temperature: float | None = None) -> Any:
    """
    Build a ChatGroq instance for cloud inference.

    Args:
        model:       Groq model name (e.g. "mixtral-8x7b-32768").
        temperature: Override the config temperature.

    Returns:
        A ChatGroq instance.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set.
    """
    from langchain_groq import ChatGroq

    cfg = get_settings()
    if not cfg.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or switch to LLM_PROVIDER=ollama."
        )
    os.environ["GROQ_API_KEY"] = cfg.groq_api_key

    llm = ChatGroq(
        model=model,
        temperature=temperature if temperature is not None else cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
    )
    logger.info("Groq LLM ready: model='%s'", model)
    return llm


def _make_openai_llm(model: str, temperature: float | None = None) -> Any:
    """
    Build a ChatOpenAI instance.

    Args:
        model:       OpenAI model name (e.g. "gpt-4o-mini").
        temperature: Override the config temperature.

    Returns:
        A ChatOpenAI instance.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set.
    """
    from langchain_openai import ChatOpenAI

    cfg = get_settings()
    if not cfg.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or switch to LLM_PROVIDER=ollama."
        )
    os.environ["OPENAI_API_KEY"] = cfg.openai_api_key

    llm = ChatOpenAI(
        model=model,
        temperature=temperature if temperature is not None else cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
    )
    logger.info("OpenAI LLM ready: model='%s'", model)
    return llm


# ---------------------------------------------------------------------------
# Health check utility
# ---------------------------------------------------------------------------

def check_ollama_connection() -> dict:
    """
    Verify that the Ollama server is reachable and list available models.

    Returns:
        A dict with keys "reachable" (bool), "models" (list[str]), "error" (str).
    """
    import requests

    cfg = get_settings()
    try:
        resp = requests.get(f"{cfg.ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        logger.info("Ollama reachable. Available models: %s", models)
        return {"reachable": True, "models": models, "error": ""}
    except Exception as exc:
        logger.warning("Ollama not reachable: %s", exc)
        return {"reachable": False, "models": [], "error": str(exc)}
