
import json
import logging
import re
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Increase output budget to reduce incomplete JSON responses.
# generate_text() must support the max_tokens parameter.
EVALUATION_MAX_TOKENS = 1200
EVALUATION_TEMPERATURE = 0.1

print("Loading interview evaluation model...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Interview evaluation model loaded!")


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "to", "of", "in", "on", "for", "with", "by", "from",
    "and", "or", "but", "if", "then", "than", "this", "that",
    "these", "those", "it", "its", "they", "their", "them",
    "can", "could", "would", "should", "may", "might", "what",
    "why", "when", "where", "which", "who", "using", "used",
    "use", "also", "have", "has", "had", "does", "did", "will"
}


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):
    text = str(text or "").lower()
    text = text.replace("-", " ").replace("/", " ")

    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9+#.\s]", " ", text)
    ).strip()


@lru_cache(maxsize=512)
def _cached_embedding(text):
    vector = embedding_model.encode(
        text,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    return vector.detach().cpu()


def calculate_similarity(text_a, text_b):
    """
    Semantic embedding similarity from 0 to 1.

    Similarity does not establish factual correctness.
    """
    text_a = str(text_a or "").strip()
    text_b = str(text_b or "").strip()

    if not text_a or not text_b:
        return 0.0

    try:
        emb_a = _cached_embedding(text_a)
        emb_b = _cached_embedding(text_b)

        similarity = torch.sum(emb_a * emb_b).item()

        return round(
            max(0.0, min(1.0, float(similarity))),
            4
        )

    except Exception:
        logger.exception("Semantic similarity calculation failed")
        return 0.0


def calculate_completeness(candidate_answer):
    """
    Weak length-based diagnostic from 0 to 1.
    Used only by fallback; length is not correctness.
    """
    words = len(str(candidate_answer or "").split())

    if words == 0:
        return 0.0

    return round(min(1.0, words / 60.0), 3)


def calculate_concept_coverage(reference_answer, candidate_answer):
    """
    Lexical overlap diagnostic from 0 to 1.
    This does not understand synonyms or prove correctness.
    """
    reference_words = {
        word.strip(".")
        for word in normalize_text(reference_answer).split()
        if len(word.strip(".")) >= 4
        and word.strip(".") not in STOP_WORDS
        and not word.strip(".").isdigit()
    }

    candidate_words = {
        word.strip(".")
        for word in normalize_text(candidate_answer).split()
    }

    if not reference_words:
        return 1.0

    matched = len(reference_words.intersection(candidate_words))

    return round(matched / len(reference_words), 3)


# ============================================================
# SCORE HELPERS
# ============================================================

def clamp_score(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(default)

    return round(max(0.0, min(100.0, value)), 1)


def validate_required_score(data, key):
    """
    Require the LLM to return a real numeric score.
    Do not silently turn missing metrics into zero.
    """
    value = data.get(key)

    if isinstance(value, bool) or value is None:
        raise ValueError(
            f"LLM evaluation is missing required score: {key}"
        )

    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"LLM returned an invalid score for {key}: {value!r}"
        ) from exc

    if not 0 <= score <= 100:
        raise ValueError(
            f"LLM score for {key} must be between 0 and 100."
        )

    return round(score, 1)


def calculate_weighted_score(
    correctness,
    concept_coverage,
    relevance,
    completeness
):
    """
    All inputs are percentages from 0 to 100.

    Correctness       40%
    Concept coverage  25%
    Relevance         20%
    Completeness      15%
    """
    score = (
        clamp_score(correctness) * 0.40
        + clamp_score(concept_coverage) * 0.25
        + clamp_score(relevance) * 0.20
        + clamp_score(completeness) * 0.15
    )

    return round(max(0.0, min(100.0, score)), 1)


def get_verdict(score):
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Partially Correct"
    if score >= 30:
        return "Needs Improvement"
    return "Incorrect"


# ============================================================
# ROBUST JSON EXTRACTION
# ============================================================

def _extract_json(response):
    """
    Extract a complete JSON object from an LLM response.

    Handles:
    - Plain JSON
    - JSON inside Markdown fences
    - Explanatory text around a JSON object

    Does not try to repair truncated JSON because that could
    silently change evaluation scores or feedback.
    """
    if isinstance(response, dict):
        return response

    if not isinstance(response, str) or not response.strip():
        raise ValueError("Empty evaluation response")

    cleaned = response.strip()

    # Remove Markdown fences if the model adds them.
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    # First try parsing the entire response.
    try:
        data = json.loads(cleaned)

        if isinstance(data, dict):
            return data

    except json.JSONDecodeError:
        pass

    # Scan for a complete JSON object, respecting quoted strings.
    for start, char in enumerate(cleaned):
        if char != "{":
            continue

        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(cleaned)):
            current = cleaned[index]

            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False

                continue

            if current == '"':
                in_string = True

            elif current == "{":
                depth += 1

            elif current == "}":
                depth -= 1

                if depth == 0:
                    candidate = cleaned[start:index + 1]

                    try:
                        data = json.loads(candidate)

                        if isinstance(data, dict):
                            return data

                    except json.JSONDecodeError:
                        break

    raise ValueError(
        "No complete valid JSON object found in evaluation response. "
        "The model may have returned empty or truncated output."
    )


def _clean_string_list(value):
    if not isinstance(value, list):
        return []

    result = []

    for item in value[:5]:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())

    return result


# ============================================================
# LLM-BASED EVALUATION
# ============================================================

def evaluate_with_llm(
    question,
    reference_answer,
    candidate_answer
):
    """
    Evaluate an answer using the project's existing generate_text()
    function from interview/question_generator.py.
    """
    try:
        from .question_generator import generate_text
    except ImportError:
        from question_generator import generate_text

    prompt = f"""
You are a fair technical interviewer evaluating an internship candidate.

Evaluate technical understanding, not exact wording. The reference answer
is guidance, not a mandatory checklist. Accept correct alternative
explanations and examples.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

CANDIDATE ANSWER:
{candidate_answer}

Give numeric percentage scores from 0 to 100:
- correctness: factual and technical accuracy.
- concept_coverage: coverage of important concepts.
- relevance: how directly the answer addresses the question.
- completeness: important details appropriate to the question.

SCORING RULES:
- Award partial credit for correct ideas.
- A concise but correct answer can score highly.
- Do not reward length or buzzwords without substance.
- Penalize genuine factual errors and important omissions.
- Judge the answer in the context of the question.
- Do not invent mistakes.
- Do not require every detail from the reference answer.
- An unrelated answer should receive a low score.

Return only one valid JSON object, with no Markdown and no extra text.
Use exactly these keys:
{{
  "correctness": 0,
  "concept_coverage": 0,
  "relevance": 0,
  "completeness": 0,
  "strengths": [],
  "missing_points": [],
  "feedback": "",
  "improved_answer": ""
}}

Each score must be a number between 0 and 100.
Strengths and missing_points must be arrays of short strings.
Feedback must be constructive. improved_answer must provide a
technically correct example answer to the question.
""".strip()

    response = generate_text(
        prompt,
        max_tokens=EVALUATION_MAX_TOKENS,
        temperature=EVALUATION_TEMPERATURE
    )

    if not isinstance(response, str):
        raise ValueError(
            "generate_text() must return a string response."
        )

    if not response.strip():
        raise ValueError("LLM returned an empty evaluation response.")

    data = _extract_json(response)

    required_keys = (
        "correctness",
        "concept_coverage",
        "relevance",
        "completeness"
    )

    for key in required_keys:
        if key not in data:
            raise ValueError(
                f"LLM response is missing required key: {key}"
            )

    correctness = validate_required_score(data, "correctness")
    coverage = validate_required_score(data, "concept_coverage")
    relevance = validate_required_score(data, "relevance")
    completeness = validate_required_score(data, "completeness")

    score = calculate_weighted_score(
        correctness=correctness,
        concept_coverage=coverage,
        relevance=relevance,
        completeness=completeness
    )

    strengths = _clean_string_list(data.get("strengths"))
    missing_points = _clean_string_list(data.get("missing_points"))

    feedback = str(data.get("feedback") or "").strip()
    improved_answer = str(data.get("improved_answer") or "").strip()

    feedback_parts = []

    if feedback:
        feedback_parts.append(feedback)

    if strengths:
        feedback_parts.append(
            "What you did well:\n- " + "\n- ".join(strengths)
        )

    if missing_points:
        feedback_parts.append(
            "What to improve:\n- " + "\n- ".join(missing_points)
        )

    if not feedback_parts:
        feedback_parts.append("LLM evaluation completed.")

    return {
        "score": score,
        "correctness": correctness,
        "concept_coverage": coverage,
        "relevance": relevance,
        "completeness": completeness,
        "verdict": get_verdict(score),
        "feedback": "\n\n".join(feedback_parts),
        "improved_answer": improved_answer,
        "evaluation_method": "groq_llm"
    }


# ============================================================
# LOW-CONFIDENCE FALLBACK
# ============================================================

def evaluate_with_fallback(
    question,
    reference_answer,
    candidate_answer
):
    """
    Fallback when LLM evaluation fails.

    Similarity, lexical coverage and length are only rough signals.
    The fallback deliberately does not claim to verify correctness.
    """
    similarity = calculate_similarity(
        reference_answer,
        candidate_answer
    )

    lexical_coverage = calculate_concept_coverage(
        reference_answer,
        candidate_answer
    )

    question_similarity = calculate_similarity(
        question,
        candidate_answer
    )

    completeness = calculate_completeness(candidate_answer)

    similarity_pct = similarity * 100
    coverage_pct = lexical_coverage * 100
    relevance_pct = question_similarity * 100
    completeness_pct = completeness * 100

    score = (
        similarity_pct * 0.30
        + coverage_pct * 0.30
        + relevance_pct * 0.30
        + completeness_pct * 0.10
    )

    # Avoid giving a substantial score to extremely short answers
    # just because their embeddings happen to be similar.
    if len(candidate_answer.split()) < 3:
        score = min(score, 20.0)

    score = round(max(0.0, min(100.0, score)), 1)

    return {
        "score": score,
        "similarity": similarity,

        # None means unverified, not a measured zero.
        "correctness": None,

        "concept_coverage": round(coverage_pct, 1),
        "completeness": round(completeness_pct, 1),
        "relevance": round(relevance_pct, 1),
        "verdict": "Low-Confidence Estimate",
        "feedback": (
            "The LLM evaluator was unavailable, so a lower-confidence "
            "similarity fallback was used. This score does not verify "
            "technical correctness. Retry the evaluation when the LLM "
            "is available."
        ),
        "improved_answer": reference_answer,
        "evaluation_method": "embedding_fallback",
        "lexical_coverage": round(coverage_pct, 1)
    }


# ============================================================
# MAIN EVALUATION ENTRY POINT
# ============================================================

def evaluate_answer(question, reference_answer, candidate_answer):
    question = str(question or "").strip()
    reference_answer = str(reference_answer or "").strip()
    candidate_answer = str(candidate_answer or "").strip()

    if not candidate_answer:
        return {
            "score": 0.0,
            "similarity": 0.0,
            "correctness": 0.0,
            "concept_coverage": 0.0,
            "completeness": 0.0,
            "relevance": 0.0,
            "verdict": "No Answer",
            "feedback": "No answer was provided.",
            "improved_answer": reference_answer,
            "evaluation_method": "no_answer"
        }

    if not question or not reference_answer:
        return {
            "score": 0.0,
            "similarity": 0.0,
            "correctness": None,
            "concept_coverage": None,
            "completeness": round(
                calculate_completeness(candidate_answer) * 100,
                1
            ),
            "relevance": None,
            "verdict": "Cannot Evaluate",
            "feedback": (
                "The question or reference answer is missing. "
                "Regenerate the question before evaluating this answer."
            ),
            "improved_answer": reference_answer,
            "evaluation_method": "missing_reference"
        }

    similarity = calculate_similarity(
        reference_answer,
        candidate_answer
    )

    lexical_coverage = calculate_concept_coverage(
        reference_answer,
        candidate_answer
    )

    try:
        result = evaluate_with_llm(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer
        )

        result["similarity"] = similarity
        result["lexical_coverage"] = round(
            lexical_coverage * 100,
            1
        )

        logger.info(
            "Answer evaluated successfully using LLM. "
            "score=%s method=%s",
            result["score"],
            result["evaluation_method"]
        )

        return result

    except Exception as exc:
        logger.exception(
            "LLM evaluation failed; using lower-confidence fallback. "
            "reason=%s",
            str(exc)
        )

        result = evaluate_with_fallback(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer
        )

        logger.warning(
            "Fallback evaluation completed. "
            "score=%s method=%s correctness=unverified",
            result["score"],
            result["evaluation_method"]
        )

        return result


# ============================================================
# OPTIONAL STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    question = input("Enter question: ").strip()
    reference_answer = input("Enter expected answer: ").strip()
    candidate_answer = input("Enter candidate answer: ").strip()

    result = evaluate_answer(
        question,
        reference_answer,
        candidate_answer
    )

    print("\n========== EVALUATION RESULT ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=======================================")
