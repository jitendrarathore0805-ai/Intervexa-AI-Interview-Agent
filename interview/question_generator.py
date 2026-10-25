
"""
AI Interview Agent - Question Generator

Primary provider : Groq GPT-OSS-20B
Fallback provider: Local Qwen
RAG support      : Resume and job-description context

Compatible with the existing Flask integration.
"""

import inspect
import logging
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = (
    os.getenv("GROQ_QUESTION_MODEL")
    or os.getenv("GROQ_MODEL")
    or "openai/gpt-oss-20b"
).strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

MAX_COMPLETION_TOKENS = max(
    128,
    int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "1600")),
)

MAX_RETRIES = max(
    1,
    min(int(os.getenv("GROQ_MAX_RETRIES", "2")), 3),
)

RETRY_MAX_WAIT = max(
    1,
    int(os.getenv("GROQ_RETRY_MAX_WAIT", "10")),
)

PRIMARY_PROVIDER = os.getenv(
    "LLM_PROVIDER", "groq"
).strip().lower()

FALLBACK_PROVIDER = os.getenv(
    "LLM_FALLBACK_PROVIDER", "local"
).strip().lower()

_client = None


# ============================================================
# CUSTOM EXCEPTION
# ============================================================

class QuestionGenerationError(RuntimeError):
    """Raised when an LLM provider cannot generate usable text."""


# ============================================================
# GROQ CLIENT
# ============================================================

def _get_client():
    """Initialize the Groq client when required."""

    global _client

    if _client is not None:
        return _client

    if not GROQ_API_KEY:
        raise QuestionGenerationError(
            "GROQ_API_KEY missing hai. Apni .env file check karo."
        )

    try:
        from groq import Groq

        _client = Groq(api_key=GROQ_API_KEY)
        return _client

    except ImportError as exc:
        raise QuestionGenerationError(
            "Groq package missing hai. Run: "
            "python -m pip install groq"
        ) from exc

    except Exception as exc:
        raise QuestionGenerationError(
            f"Groq client initialize nahi hua: {exc}"
        ) from exc


def _is_gpt_oss(model_name: str) -> bool:
    return "gpt-oss" in model_name.lower()


# ============================================================
# RESPONSE EXTRACTION
# ============================================================

def _extract_response_text(response: Any) -> str:
    """Safely extract text from a Groq completion."""

    choices = getattr(response, "choices", None)

    if not choices:
        return ""

    message = getattr(choices[0], "message", None)

    if message is None:
        return ""

    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                value = item.get("text")
            else:
                value = getattr(item, "text", None)

            if value:
                parts.append(str(value))

        return "\n".join(parts).strip()

    return ""


# ============================================================
# GROQ GENERATION
# ============================================================

def _generate_groq_text(
    prompt: str,
    max_tokens: int = 400,
    temperature: float = 0.3,
    system_prompt: Optional[str] = None,
) -> str:
    """Generate concise text using Groq."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise QuestionGenerationError("Prompt empty hai.")

    client = _get_client()

    requested_tokens = max(64, int(max_tokens))
    token_limit = min(
        requested_tokens,
        MAX_COMPLETION_TOKENS,
    )

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt.strip(),
        })

    messages.append({
        "role": "user",
        "content": prompt.strip(),
    })

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            options = {
                "model": MODEL_NAME,
                "messages": messages,
                "max_completion_tokens": token_limit,
                "temperature": temperature,
                "stream": False,
            }

            if _is_gpt_oss(MODEL_NAME):
                options["reasoning_effort"] = "low"

            response = client.chat.completions.create(**options)

            choices = getattr(response, "choices", None)

            finish_reason = (
                getattr(choices[0], "finish_reason", None)
                if choices
                else None
            )

            text = _extract_response_text(response)

            logger.info(
                "Groq model=%s attempt=%s/%s "
                "finish_reason=%s token_budget=%s",
                MODEL_NAME,
                attempt,
                MAX_RETRIES,
                finish_reason,
                token_limit,
            )

            # A truncated response must not be treated as complete.
            if finish_reason == "length":
                raise QuestionGenerationError(
                    "Groq response token limit par truncate hua."
                )

            if not text:
                raise QuestionGenerationError(
                    "Groq ne empty response return kiya."
                )

            return text

        except Exception as exc:
            last_error = exc

            logger.warning(
                "Groq attempt %s/%s failed: %s",
                attempt,
                MAX_RETRIES,
                exc,
            )

            if attempt < MAX_RETRIES:
                time.sleep(
                    min(RETRY_MAX_WAIT, 2 ** (attempt - 1))
                )

    raise QuestionGenerationError(
        f"Groq generation failed: {last_error}"
    )


# ============================================================
# QWEN LOCAL FALLBACK
# ============================================================

def _call_qwen_compatibly(
    qwen_generate_text,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: Optional[str],
) -> str:
    """Support different argument names in the existing Qwen loader."""

    try:
        signature = inspect.signature(qwen_generate_text)
        parameters = signature.parameters

        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in parameters.values()
        )

    except (TypeError, ValueError):
        parameters = {}
        accepts_kwargs = True

    kwargs = {}

    if "max_new_tokens" in parameters:
        kwargs["max_new_tokens"] = max_tokens
    elif "max_tokens" in parameters:
        kwargs["max_tokens"] = max_tokens
    elif accepts_kwargs:
        kwargs["max_new_tokens"] = max_tokens

    if "temperature" in parameters or accepts_kwargs:
        kwargs["temperature"] = temperature

    if "system_prompt" in parameters or accepts_kwargs:
        kwargs["system_prompt"] = system_prompt

    result = qwen_generate_text(prompt, **kwargs)

    return str(result).strip() if result is not None else ""


def _generate_local_text(
    prompt: str,
    max_tokens: int = 400,
    temperature: float = 0.3,
    system_prompt: Optional[str] = None,
) -> str:
    """Generate text with the existing local Qwen model."""

    try:
        # Lazy import prevents loading Qwen unless it is needed.
        from models.model_loader import generate_text as qwen_generate_text

        result = _call_qwen_compatibly(
            qwen_generate_text=qwen_generate_text,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )

        if not result:
            raise QuestionGenerationError(
                "Local Qwen ne empty response return kiya."
            )

        return result

    except QuestionGenerationError:
        raise

    except Exception as exc:
        logger.exception("Qwen generation failed")

        raise QuestionGenerationError(
            f"Qwen generation failed: {exc}"
        ) from exc


# ============================================================
# MAIN PROVIDER ROUTER
# ============================================================

def generate_text(
    prompt: str,
    max_tokens: int = 400,
    temperature: float = 0.3,
    system_prompt: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Shared generation function for Flask and evaluator.

    Groq is tried first by default; Qwen is the fallback.
    """

    if kwargs.get("max_new_tokens") is not None:
        max_tokens = kwargs["max_new_tokens"]

    if not isinstance(prompt, str) or not prompt.strip():
        raise QuestionGenerationError("Prompt empty hai.")

    providers = []

    for provider in (PRIMARY_PROVIDER, FALLBACK_PROVIDER):
        if provider in ("groq", "local") and provider not in providers:
            providers.append(provider)

    if not providers:
        providers = ["groq", "local"]

    errors = []

    for provider in providers:
        try:
            logger.info("Trying LLM provider: %s", provider)

            if provider == "groq":
                result = _generate_groq_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt,
                )
            else:
                result = _generate_local_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt,
                )

            if result:
                logger.info(
                    "LLM provider succeeded: %s",
                    provider,
                )
                return result.strip()

            raise QuestionGenerationError(
                f"Provider {provider} returned an empty response."
            )

        except Exception as exc:
            errors.append(f"{provider}: {exc}")

            logger.warning(
                "Provider %s failed; checking fallback: %s",
                provider,
                exc,
            )

    raise QuestionGenerationError(
        "All LLM providers failed. " + " | ".join(errors)
    )


# ============================================================
# CONTEXT HELPERS
# ============================================================

def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text if text else default


def _context_text(context: Any) -> str:
    """Convert retrieved RAG context into readable text."""

    if context is None:
        return ""

    if isinstance(context, str):
        return context.strip()

    if isinstance(context, dict):
        content = (
            context.get("page_content")
            or context.get("content")
            or context.get("text")
        )
        return str(content or "").strip()

    if isinstance(context, (list, tuple)):
        parts = []

        for item in context:
            if isinstance(item, str):
                parts.append(item.strip())

            elif isinstance(item, dict):
                content = (
                    item.get("page_content")
                    or item.get("content")
                    or item.get("text")
                )

                if content:
                    parts.append(str(content).strip())

            else:
                content = getattr(item, "page_content", None)

                if content:
                    parts.append(str(content).strip())

        return "\n\n".join(part for part in parts if part)

    content = getattr(context, "page_content", None)

    return str(content or "").strip()


def _skills_text(skills: Any, default_skill: str) -> str:
    if skills is None:
        return default_skill

    if isinstance(skills, (list, tuple, set)):
        return ", ".join(str(item) for item in skills)

    return str(skills).strip()


# ============================================================
# SHORT INTERVIEW QUESTION
# ============================================================

def generate_question(
    role: str = "Software Engineer",
    skill: str = "Machine Learning",
    difficulty: str = "Medium",
    context: Any = None,
    previous_questions: Optional[list] = None,
    interview_type: str = "Technical",
    skills: Any = None,
    job_description: str = "",
    **kwargs,
) -> str:
    """Generate one short interview question."""

    role = _clean_text(role, "Software Engineer")
    skill = _clean_text(skill, "Machine Learning")
    difficulty = _clean_text(difficulty, "Medium")
    interview_type = _clean_text(interview_type, "Technical")

    skills_text = _skills_text(skills, skill)

    previous_questions = previous_questions or []

    previous_text = "\n".join(
        f"- {str(item)}"
        for item in previous_questions[-8:]
    )

    rag_context = _context_text(context)[:2000]
    job_description = _clean_text(job_description)[:1200]

    prompt = f"""
Generate exactly ONE short interview question.

Role: {role}
Interview type: {interview_type}
Topic: {skill}
Skills: {skills_text}
Difficulty: {difficulty}

Job description:
{job_description or "Not provided"}

Relevant resume or job context:
{rag_context or "Not provided"}

Previously asked questions:
{previous_text or "None"}

Rules:
- Ask only one question.
- Keep it under 25 words.
- Use clear, natural English.
- Match the selected role and difficulty.
- Avoid repeating earlier questions.
- Do not include the answer or an explanation.
- Return only the question.
""".strip()

    question = generate_text(
        prompt,
        max_tokens=150,
        temperature=0.3,
        system_prompt=(
            "You are a professional interviewer. "
            "Generate one short, realistic interview question."
        ),
    )

    question = question.strip().strip('"').strip()

    if not question:
        raise QuestionGenerationError(
            "Question generator returned an empty question."
        )

    return question


# ============================================================
# SHORT EXPECTED ANSWER
# ============================================================

def generate_expected_answer(
    question: str,
    role: str = "Software Engineer",
    skill: str = "Machine Learning",
    difficulty: str = "Medium",
    context: Any = None,
    skills: Any = None,
    job_description: str = "",
    interview_type: str = "Technical",
    **kwargs,
) -> str:
    """Generate a concise reference answer of about 50-80 words."""

    question = _clean_text(question)

    if not question:
        raise QuestionGenerationError(
            "Expected answer ke liye question empty hai."
        )

    role = _clean_text(role, "Software Engineer")
    skill = _clean_text(skill, "Machine Learning")
    difficulty = _clean_text(difficulty, "Medium")
    interview_type = _clean_text(interview_type, "Technical")

    skills_text = _skills_text(skills, skill)

    rag_context = _context_text(context)[:2000]
    job_description = _clean_text(job_description)[:1200]

    prompt = f"""
Write a short, accurate reference answer for an interview.

Role: {role}
Interview type: {interview_type}
Topic: {skill}
Skills: {skills_text}
Difficulty: {difficulty}

Question:
{question}

Job description:
{job_description or "Not provided"}

Relevant context:
{rag_context or "Not provided"}

Rules:
- Use simple, professional English.
- Aim for 50-80 words.
- Answer the question directly.
- Explain the main concept accurately.
- Add one brief example only if useful.
- Avoid unnecessary details and long introductions.
- Do not invent resume facts.
- Return only the answer.
""".strip()

    answer = generate_text(
        prompt,
        max_tokens=400,
        temperature=0.2,
        system_prompt=(
            "You are an experienced interview coach. "
            "Write concise, accurate reference answers."
        ),
    )

    answer = answer.strip()

    if not answer:
        raise QuestionGenerationError(
            "Expected-answer generator returned an empty answer."
        )

    return answer


# ============================================================
# FLASK INTEGRATION
# ============================================================

def generate_interview_content(
    role: str = "Software Engineer",
    skill: str = "Machine Learning",
    difficulty: str = "Medium",
    context: Any = None,
    previous_questions: Optional[list] = None,
    interview_type: str = "Technical",
    skills: Any = None,
    job_description: str = "",
    **kwargs,
) -> dict:
    """
    Flask integration: returns a question and expected answer.
    """

    question = generate_question(
        role=role,
        skill=skill,
        difficulty=difficulty,
        context=context,
        previous_questions=previous_questions,
        interview_type=interview_type,
        skills=skills,
        job_description=job_description,
    )

    expected_answer = generate_expected_answer(
        question=question,
        role=role,
        skill=skill,
        difficulty=difficulty,
        context=context,
        skills=skills,
        job_description=job_description,
        interview_type=interview_type,
    )

    return {
        "question": question,
        "expected_answer": expected_answer,
        "source": "local_llm",
    }


# ============================================================
# BACKWARD-COMPATIBILITY ALIASES
# ============================================================

generate_interview_question = generate_question
generate_model_answer = generate_expected_answer
generate_reference_answer = generate_expected_answer
