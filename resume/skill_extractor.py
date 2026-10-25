
"""Lightweight, explainable resume skill and project extraction."""
import re

SKILL_ALIASES = {
    "python": [r"\bpython\b"],
    "sql": [r"\bsql\b", r"\bmysql\b", r"\bpostgres(?:ql)?\b", r"\bsqlite\b"],
    "machine learning": [r"\bmachine learning\b", r"\bscikit[-.]learn\b", r"\bsklearn\b"],
    "deep learning": [r"\bdeep learning\b"],
    "nlp": [r"\bnlp\b", r"\bnatural language processing\b"],
    "generative ai": [r"\bgenerative ai\b", r"\bgenai\b", r"\bllms?\b", r"\blarge language models?\b"],
    "rag": [r"\brag\b", r"\bretrieval[- ]augmented generation\b"],
    "tensorflow": [r"\btensorflow\b"],
    "pytorch": [r"\bpytorch\b", r"\btorch\b"],
    "keras": [r"\bkeras\b"],
    "scikit-learn": [r"\bscikit[-.]learn\b", r"\bsklearn\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
    "opencv": [r"\bopencv\b", r"\bcv2\b"],
    "computer vision": [r"\bcomputer vision\b", r"\bimage classification\b", r"\bobject detection\b"],
    "flask": [r"\bflask\b"],
    "django": [r"\bdjango\b"],
    "fastapi": [r"\bfastapi\b"],
    "rest api": [r"\brestful? apis?\b", r"\bapi development\b"],
    "html": [r"\bhtml5?\b"],
    "css": [r"\bcss3?\b"],
    "javascript": [r"\bjavascript\b", r"\bjs\b"],
    "react": [r"\breact(?:\.js)?\b"],
    "git": [r"\bgit\b", r"\bgithub\b"],
    "docker": [r"\bdocker\b"],
    "aws": [r"\baws\b", r"\bamazon web services\b"],
    "azure": [r"\bazure\b"],
    "linux": [r"\blinux\b"],
    "excel": [r"\bexcel\b", r"\bmicrosoft excel\b"],
    "statistics": [r"\bstatistics\b", r"\bstatistical analysis\b"],
    "data analysis": [r"\bdata analysis\b", r"\bdata analytics\b"],
    "data visualization": [
        r"\bdata visualization\b", r"\btableau\b", r"\bpower bi\b",
        r"\bmatplotlib\b", r"\bseaborn\b"
    ],
    "linear regression": [r"\blinear regression\b"],
    "neural networks": [r"\bneural networks?\b", r"\bartificial neural networks?\b"],
    "cnn": [r"\bcnn\b", r"\bconvolutional neural networks?\b"],
    "rnn": [r"\brnn\b", r"\brecurrent neural networks?\b"],
    "transformers": [r"\btransformers?\b", r"\bbert\b", r"\bgpt\b"],
    "prompt engineering": [r"\bprompt engineering\b"],
    "vector databases": [
        r"\bvector databases?\b", r"\bchromadb\b", r"\bfaiss\b",
        r"\bqdrant\b", r"\bweaviate\b"
    ],
    "langchain": [r"\blangchain\b"],
    "langgraph": [r"\blanggraph\b"],
    "mongodb": [r"\bmongodb\b"],
    "powerpoint": [r"\bpowerpoint\b", r"\bppt\b"],
    "communication": [r"\bcommunication skills?\b"],
    "electrical engineering": [r"\belectrical engineering\b"],
    "electrical system design": [r"\belectrical systems? design\b"],
    "energy efficiency": [r"\benergy efficiency\b"],
    "requirements analysis": [r"\brequirements analysis\b", r"\bsystem requirements\b"],
    "project management": [r"\bproject management\b", r"\bproject execution\b"],
    "teamwork": [r"\bteamwork\b", r"\bcross[- ]functional teams?\b"],
}

PROJECT_HEADERS = re.compile(
    r"^\s*(?:academic\s+projects?|personal\s+projects?|key\s+projects?|projects?)\s*:?\s*$",
    re.I
)

SECTION_HEADERS = re.compile(
    r"^\s*(?:education|academic background|certifications?|certificates|"
    r"technical skills|skills|achievements|positions of responsibility|"
    r"languages|experience|work experience|professional experience|"
    r"employment history|internships?|summary|objective)\s*:?\s*$",
    re.I
)

EXPERIENCE_HEADERS = re.compile(
    r"^\s*(?:work experience|professional experience|experience|"
    r"employment history|internships?)\s*:?\s*$",
    re.I
)

EDUCATION_HEADERS = re.compile(
    r"^\s*(?:education|academic background|educational qualifications?)\s*:?\s*$",
    re.I
)

CERTIFICATION_HEADERS = re.compile(
    r"^\s*(?:certifications?|certificates)\s*:?\s*$",
    re.I
)


def _clean_lines(text):
    """Normalize extracted text while preserving line boundaries."""
    text = (text or "").replace("\x00", " ")
    lines = []

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line)
        line = line.strip(" \t•●▪◦-–")
        if line:
            lines.append(line)

    return lines


def extract_skills(text, extra_skills=None):
    """Return recognized skills that are explicitly present in the text."""
    text = text or ""
    found = []
    seen = set()

    for canonical, patterns in SKILL_ALIASES.items():
        if any(re.search(pattern, text, re.I) for pattern in patterns):
            key = canonical.casefold()
            if key not in seen:
                found.append(canonical)
                seen.add(key)

    # Add custom skills only when their text actually appears in the resume.
    for skill in extra_skills or []:
        skill = str(skill).strip()
        if not skill:
            continue

        if re.search(r"(?<!\w)" + re.escape(skill) + r"(?!\w)", text, re.I):
            key = skill.casefold()
            if key not in seen:
                found.append(skill)
                seen.add(key)

    return found


def _extract_section(text, header_patterns, stop_patterns=None, max_lines=12):
    """Extract lines under a heading until the next recognized section."""
    lines = _clean_lines(text)
    stop_patterns = stop_patterns or []
    active = False
    result = []

    for line in lines:
        if header_patterns.match(line):
            active = True
            continue

        if not active:
            continue

        if any(pattern.match(line) for pattern in stop_patterns):
            break

        # Ignore an unrelated all-caps heading where possible.
        if (
            len(line) <= 50
            and line.isupper()
            and re.search(r"[A-Z]", line)
            and not re.search(r"\d", line)
        ):
            break

        result.append(line[:350])

        if len(result) >= max_lines:
            break

    return result


def extract_projects(text, max_projects=8):
    """
    Extract only content under an explicit Projects heading.

    Does not use experience descriptions as project fallbacks.
    """
    lines = _clean_lines(text)
    captured = []
    in_projects = False

    for line in lines:
        if PROJECT_HEADERS.match(line):
            in_projects = True
            continue

        if not in_projects:
            continue

        if SECTION_HEADERS.match(line):
            break

        if len(line) >= 8 and line not in captured:
            captured.append(line[:350])

        if len(captured) >= max_projects:
            break

    return captured


def analyze_resume(text):
    """Return normalized resume analysis fields."""
    text = text or ""

    education_headers = re.compile(
        r"^\s*(?:education|academic background|educational qualifications?)\s*:?\s*$",
        re.I
    )
    experience_headers = re.compile(
        r"^\s*(?:work experience|professional experience|experience|"
        r"employment history|internships?)\s*:?\s*$",
        re.I
    )
    certification_headers = re.compile(
        r"^\s*(?:certifications?|certificates)\s*:?\s*$",
        re.I
    )

    stop_patterns = [
        PROJECT_HEADERS,
        SECTION_HEADERS,
    ]

    normalized_text = re.sub(r"\s+", " ", text).strip()

    return {
        "skills": extract_skills(text),
        "projects": extract_projects(text),
        "education": _extract_section(
            text, education_headers, stop_patterns, max_lines=8
        ),
        "experience": _extract_section(
            text, experience_headers, stop_patterns, max_lines=10
        ),
        "certifications": _extract_section(
            text, certification_headers, stop_patterns, max_lines=8
        ),
        "summary": normalized_text[:1200],
    }
