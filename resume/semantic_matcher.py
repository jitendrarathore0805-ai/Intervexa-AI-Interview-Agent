
"""Explainable resume-to-job matching with optional local embeddings."""
import re

from .skill_extractor import extract_skills
from .jd_analyzer import analyze_job_description


def _cosine_similarity(resume_text, job_description):
    """
    Return (similarity, method).
    Falls back to lexical similarity when embeddings are unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        vectors = model.encode(
            [resume_text, job_description],
            normalize_embeddings=True,
        )

        import numpy as np

        similarity = float(np.dot(vectors[0], vectors[1]))
        similarity = max(0.0, min(1.0, similarity))

        return round(similarity, 4), "sentence-transformers/all-MiniLM-L6-v2"

    except Exception:
        # Explainable fallback: compare normalized words.
        resume_words = set(
            re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.-]*\b", resume_text.casefold())
        )
        job_words = set(
            re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.-]*\b", job_description.casefold())
        )

        stop_words = {
            "the", "and", "for", "with", "that", "this", "you", "your",
            "are", "our", "will", "have", "has", "from", "who", "what",
            "job", "role", "work", "years", "year", "required", "preferred",
            "candidate", "experience", "ability", "strong",
        }

        resume_words -= stop_words
        job_words -= stop_words

        if not resume_words or not job_words:
            return None, "lexical_fallback"

        # Jaccard similarity is a transparent lexical overlap estimate.
        score = len(resume_words & job_words) / len(resume_words | job_words)
        return round(score, 4), "lexical_fallback"


def _unique_skills(skills):
    """Deduplicate skills without changing their display names."""
    result = []
    seen = set()

    for skill in skills or []:
        value = str(skill).strip()
        key = value.casefold()

        if value and key not in seen:
            result.append(value)
            seen.add(key)

    return result


def match_resume_to_job(
    resume_text,
    job_description,
    resume_skills=None,
    job_skills=None,
):
    """Calculate skill overlap and optional text similarity."""
    resume_text = (resume_text or "").strip()
    job_description = (job_description or "").strip()

    rskills = _unique_skills(
        resume_skills if resume_skills is not None
        else extract_skills(resume_text)
    )

    if job_skills is not None:
        jskills = _unique_skills(job_skills)
    elif job_description:
        try:
            jd_result = analyze_job_description(job_description) or {}
            jskills = _unique_skills(jd_result.get("skills", []))
        except Exception:
            jskills = extract_skills(job_description)
    else:
        jskills = []

    resume_skill_map = {skill.casefold(): skill for skill in rskills}
    job_skill_map = {skill.casefold(): skill for skill in jskills}

    matched = [
        job_skill_map[key]
        for key in job_skill_map
        if key in resume_skill_map
    ]

    missing = [
        job_skill_map[key]
        for key in job_skill_map
        if key not in resume_skill_map
    ]

    if jskills:
        skill_score = round(100 * len(matched) / len(jskills))
    else:
        skill_score = None

    semantic_score = None
    method = "not_available"

    if resume_text and job_description:
        semantic_score, method = _cosine_similarity(
            resume_text[:8000],
            job_description[:8000],
        )

    if skill_score is None and semantic_score is None:
        overall = None
    elif skill_score is None:
        overall = round(semantic_score * 100)
    elif semantic_score is None:
        overall = skill_score
    else:
        overall = round(
            0.65 * skill_score + 0.35 * semantic_score * 100
        )

    return {
        "match_score": overall,
        "skill_match_score": skill_score,
        "semantic_similarity": semantic_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "resume_skills": rskills,
        "job_skills": jskills,
        "method": method,
        "note": (
            "Heuristic fit estimate, not a hiring decision. "
            "Results depend on PDF text extraction, recognized skill vocabulary, "
            "job-description quality, and the selected similarity method."
        ),
    }
