# src/reasoning_chain.py
"""
Reasoning and explanation chain for the Agentic RAG system.

This module provides structured chain-of-thought (CoT) reasoning and
explanation generation on top of retrieved answers. It adds transparency
by showing the step-by-step reasoning process before presenting the final answer.

Features:
    - Step-by-step reasoning trace (ReAct-style inner monologue).
    - Structured JSON output with reasoning steps + final answer.
    - Multilingual support (Arabic + English reasoning output).
    - Confidence estimation for the generated answer.
    - Source attribution summary from retrieved context.
    - Explanation simplification (ELI5-style) for complex topics.

Reasoning schema (ReasonedResponse):
    {
        "question":        str,          # original user question
        "language":        str,          # detected language ("ar" / "en" / "mixed")
        "reasoning_steps": List[str],    # numbered CoT steps
        "answer":          str,          # final concise answer
        "confidence":      float,        # 0.0–1.0 self-assessed confidence
        "sources_used":    List[str],    # source snippets referenced
        "explanation":     str,          # plain-language explanation (optional)
    }
"""

import json
import logging
import re
from typing import Any, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.arabic_utils import detect_language
from src.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ReasonedResponse(BaseModel):
    """
    Structured response that includes transparent reasoning steps.

    Attributes:
        question:        The original user question.
        language:        Detected primary language of the question.
        reasoning_steps: Ordered list of reasoning steps leading to the answer.
        answer:          The final answer to the question.
        confidence:      Self-assessed confidence score (0.0 = uncertain, 1.0 = certain).
        sources_used:    Key excerpts from retrieved context that informed the answer.
        explanation:     Optional plain-language explanation of the answer.
    """

    question: str
    language: str = "en"
    reasoning_steps: List[str] = Field(default_factory=list)
    answer: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    sources_used: List[str] = Field(default_factory=list)
    explanation: str = ""

    def format_for_display(self) -> str:
        """
        Render the response as a clean, human-readable string.

        Uses RTL markers for Arabic text and numbered reasoning steps.

        Returns:
            A formatted multi-section string ready for terminal or UI display.
        """
        is_arabic = self.language == "ar"
        sep = "─" * 60

        if is_arabic:
            sections = [
                f"\n{sep}",
                f"🤔 **خطوات التفكير:**",
            ]
            for i, step in enumerate(self.reasoning_steps, 1):
                sections.append(f"  {i}. {step}")
            sections += [
                f"\n📝 **الإجابة:**\n{self.answer}",
                f"\n💡 **الشرح:**\n{self.explanation}" if self.explanation else "",
                f"\n📊 **مستوى الثقة:** {self.confidence * 100:.0f}%",
            ]
        else:
            sections = [
                f"\n{sep}",
                "🤔 **Reasoning Steps:**",
            ]
            for i, step in enumerate(self.reasoning_steps, 1):
                sections.append(f"  {i}. {step}")
            sections += [
                f"\n📝 **Answer:**\n{self.answer}",
                f"\n💡 **Explanation:**\n{self.explanation}" if self.explanation else "",
                f"\n📊 **Confidence:** {self.confidence * 100:.0f}%",
            ]

        if self.sources_used:
            label = "📚 **المصادر المستخدمة:**" if is_arabic else "📚 **Sources Referenced:**"
            sections.append(f"\n{label}")
            for src in self.sources_used[:3]:  # Show max 3
                sections.append(f"  • {src[:120]}...")

        sections.append(sep)
        return "\n".join(s for s in sections if s)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_REASONING_SYSTEM_EN = """You are an expert analytical assistant. Your task is to answer the user's question using the provided context, while showing your full reasoning process.

You MUST respond with a valid JSON object following this exact schema:
{
  "reasoning_steps": [
    "Step 1: Analyze what the question is asking...",
    "Step 2: Review the relevant parts of the context...",
    "Step 3: Synthesize the information...",
    "Step 4: Formulate the answer..."
  ],
  "answer": "The concise, direct answer to the question.",
  "confidence": 0.85,
  "sources_used": ["Brief excerpt from context that was used..."],
  "explanation": "A plain-language explanation of the answer for someone unfamiliar with the topic."
}

Rules:
- reasoning_steps: 3–6 numbered steps showing your actual thought process.
- answer: Direct, complete, grounded only in the provided context.
- confidence: Float between 0.0 and 1.0. Be honest—lower it if context is ambiguous.
- sources_used: 1–3 short excerpts from context that directly informed your answer.
- explanation: 2–4 sentences in plain language. Avoid jargon.
- If context does not contain enough information, say so explicitly in the answer.
- Do NOT hallucinate or use knowledge outside the provided context.
- Return ONLY the JSON object. No markdown fences, no preamble."""

_REASONING_SYSTEM_AR = """أنت مساعد تحليلي خبير. مهمتك هي الإجابة على سؤال المستخدم باستخدام السياق المقدم مع إظهار عملية تفكيرك الكاملة.

يجب أن تجيب بكائن JSON صالح وفق هذا النموذج بالضبط:
{
  "reasoning_steps": [
    "الخطوة 1: تحليل ما يطرحه السؤال...",
    "الخطوة 2: مراجعة الأجزاء ذات الصلة من السياق...",
    "الخطوة 3: تجميع المعلومات...",
    "الخطوة 4: صياغة الإجابة..."
  ],
  "answer": "الإجابة المباشرة والموجزة على السؤال.",
  "confidence": 0.85,
  "sources_used": ["مقتطف موجز من السياق الذي تم استخدامه..."],
  "explanation": "شرح بلغة بسيطة للإجابة."
}

القواعد:
- reasoning_steps: 3-6 خطوات مرقمة تُظهر عملية تفكيرك الفعلية.
- answer: إجابة مباشرة وكاملة مستندة فقط إلى السياق المقدم.
- confidence: رقم عشري بين 0.0 و 1.0. كن صادقاً—قلّله إن كان السياق غامضاً.
- sources_used: 1-3 مقتطفات قصيرة من السياق أثّرت مباشرة في إجابتك.
- explanation: 2-4 جمل بلغة بسيطة. تجنب المصطلحات التقنية.
- إذا لم يتضمن السياق معلومات كافية، صرّح بذلك صراحةً في الإجابة.
- لا تخترع معلومات خارج السياق المقدم.
- أعد كائن JSON فقط. بدون علامات markdown، بدون مقدمة."""


# ---------------------------------------------------------------------------
# Core reasoning chain
# ---------------------------------------------------------------------------

def build_reasoning_prompt(
    question: str,
    context_chunks: List[str],
    language: str = "en",
) -> List[Any]:
    """
    Construct the message list for the reasoning LLM call.

    Args:
        question:       The user's question.
        context_chunks: Retrieved context documents.
        language:       "ar" or "en" — selects the prompt language.

    Returns:
        A list of [SystemMessage, HumanMessage] for the LLM.
    """
    system_prompt = _REASONING_SYSTEM_AR if language == "ar" else _REASONING_SYSTEM_EN
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "No context available."

    if language == "ar":
        human_text = f"السياق:\n{context_text}\n\nالسؤال: {question}"
    else:
        human_text = f"Context:\n{context_text}\n\nQuestion: {question}"

    return [SystemMessage(content=system_prompt), HumanMessage(content=human_text)]


def _parse_reasoning_response(raw: str, question: str, language: str) -> ReasonedResponse:
    """
    Parse the LLM's JSON response into a ReasonedResponse model.

    Handles common formatting issues (markdown fences, trailing commas).

    Args:
        raw:      Raw string output from the LLM.
        question: Original question (for fallback population).
        language: Detected language.

    Returns:
        A populated ReasonedResponse instance.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Remove JavaScript-style trailing commas
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        data = json.loads(cleaned)
        return ReasonedResponse(
            question=question,
            language=language,
            reasoning_steps=data.get("reasoning_steps", []),
            answer=data.get("answer", raw),
            confidence=float(data.get("confidence", 0.5)),
            sources_used=data.get("sources_used", []),
            explanation=data.get("explanation", ""),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse reasoning JSON (%s); using raw answer.", exc)
        return ReasonedResponse(
            question=question,
            language=language,
            reasoning_steps=["Direct answer generated (structured reasoning unavailable)."],
            answer=raw,
            confidence=0.4,
            sources_used=[],
            explanation="",
        )


def generate_reasoned_response(
    question: str,
    context_chunks: List[str],
    llm: Any,
    language: Optional[str] = None,
) -> ReasonedResponse:
    """
    Generate a fully reasoned, explained response for the given question.

    This is the primary entry-point for the reasoning feature. It:
    1. Detects the language of the question.
    2. Calls the LLM with a structured CoT prompt.
    3. Parses the JSON reasoning output.
    4. Returns a ReasonedResponse with steps, answer, confidence, and explanation.

    Args:
        question:       The user's question.
        context_chunks: Retrieved documents used as grounding context.
        llm:            A LangChain-compatible chat model.
        language:       Force a language ("ar" / "en"). Auto-detects if None.

    Returns:
        A ReasonedResponse instance.
    """
    cfg = get_settings()

    # Detect or use configured language
    if language is None:
        if cfg.reasoning_language == "auto":
            language = detect_language(question)
        else:
            language = cfg.reasoning_language

    logger.info(
        "Generating reasoned response | language=%s | context_chunks=%d",
        language, len(context_chunks),
    )

    messages = build_reasoning_prompt(question, context_chunks, language)

    try:
        response = llm.invoke(messages)
        raw_content = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.error("Reasoning LLM call failed: %s", exc)
        return ReasonedResponse(
            question=question,
            language=language,
            reasoning_steps=["LLM call failed; cannot generate reasoning trace."],
            answer=f"Error generating reasoned response: {exc}",
            confidence=0.0,
        )

    return _parse_reasoning_response(raw_content, question, language)


# ---------------------------------------------------------------------------
# Explanation-only helper
# ---------------------------------------------------------------------------

def explain_answer(answer: str, question: str, llm: Any, language: str = "en") -> str:
    """
    Generate a plain-language ELI5-style explanation for an existing answer.

    Useful when `generate_reasoned_response` is not used but a simple
    explanation of a raw answer is needed.

    Args:
        answer:   The answer text to explain.
        question: The original question for context.
        llm:      A LangChain-compatible chat model.
        language: "ar" or "en".

    Returns:
        A simplified explanation string.
    """
    if language == "ar":
        prompt = (
            f"السؤال: {question}\n"
            f"الإجابة: {answer}\n\n"
            "اشرح هذه الإجابة بلغة بسيطة ومفهومة لشخص عادي في 3-4 جمل فقط."
        )
    else:
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "Explain this answer in simple, plain language for a non-expert in 3-4 sentences."
        )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as exc:
        logger.warning("Explanation generation failed: %s", exc)
        return ""
