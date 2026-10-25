
"""Explainable job-description skill extraction using local regex rules."""

import re

from .skill_extractor import extract_skills


RESPONSIBILITY_HEADERS = {
    "responsibilities",
    "key responsibilities",
    "what you will do",
    "what you'll do",
    "roles and responsibilities",
    "duties",
    "job responsibilities",
}

QUALIFICATION_HEADERS = {
    "qualifications",
    "requirements",
    "required qualifications",
    "required skills",
    "what we're looking for",
    "what we are looking for",
    "preferred qualifications",
    "eligibility",
    "skills and qualifications",
}

OTHER_HEADERS = {
    "about the role",
    "about the job",
    "job description",
    "job summary",
    "overview",
    "benefits",
    "about us",
    "company overview",
    "what we offer",
    "equal opportunity",
}


def _normalize_heading(line):
    """Normalize a possible section heading."""
    return re.sub(r"\s+", " ", line).strip().strip(":").casefold()


def _clean_lines(text):
    """Normalize lines while preserving the section structure."""
    result = []

    for raw_line in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        line = line.strip(" \t•●▪◦*-–")

        if line:
            result.append(line)

    return result


def analyze_job_description(job_description):
    """
    Extract skills, responsibilities, qualifications, and summary.

    Skills are identified using the shared skill vocabulary in
    resume.skill_extractor. This is a heuristic, not an exhaustive parser.
    """
    text = (job_description or "").strip()

    if not text:
        return {
            "skills": [],
            "responsibilities": [],
            "qualifications": [],
            "summary": "",
        }

    skills = extract_skills(text)
    lines = _clean_lines(text)

    responsibilities = []
    qualifications = []
    active_section = None

    for line in lines:
        heading = _normalize_heading(line)

        if heading in RESPONSIBILITY_HEADERS:
            active_section = "responsibilities"
            continue

        if heading in QUALIFICATION_HEADERS:
            active_section = "qualifications"
            continue

        if heading in OTHER_HEADERS:
            active_section = None
            continue

        # Recognize common headings that include a colon, such as
        # "Technical Requirements:" or "Responsibilities and Duties:".
        if re.fullmatch(
            r"(?:technical|minimum|basic|preferred)\s+"
            r"(?:skills|requirements|qualifications)",
            heading,
        ):
            active_section = "qualifications"
            continue

        if active_section == "responsibilities":
            if len(line) >= 12 and line not in responsibilities:
                responsibilities.append(line[:350])

        elif active_section == "qualifications":
            if len(line) >= 8 and line not in qualifications:
                qualifications.append(line[:350])

    return {
        "skills": skills,
        "responsibilities": responsibilities[:12],
        "qualifications": qualifications[:12],
        "summary": re.sub(r"\s+", " ", text).strip()[:1200],
    }
