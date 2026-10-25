
import json
import logging
import os
import re
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.utils import secure_filename

from database.db import db
from database.models import InterviewSession, InterviewQuestion

from rag.vector_store import index_resume_text
from rag.retriever import retrieve_context

from interview.question_generator import generate_interview_content
from interview.evaluator import evaluate_answer

from resume.resume_parser import parse_resume, ResumeParseError
from resume.skill_extractor import analyze_resume
from resume.semantic_matcher import match_resume_to_job

try:
    from resume.jd_analyzer import analyze_job_description
except ImportError:
    analyze_job_description = None


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY",
    "change-this-secret-key-before-deployment"
)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///interview.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

db.init_app(app)

TOTAL_INTERVIEW_QUESTIONS = 5
ALLOWED_RESUME_EXTENSIONS = {".pdf", ".docx", ".txt"}

RESUME_DATA_DIR = Path(app.instance_path) / "resume_data"
RESUME_DATA_DIR.mkdir(parents=True, exist_ok=True)

with app.app_context():
    db.create_all()


# ============================================================
# GENERAL HELPERS
# ============================================================

def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def value_to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return "; ".join(
            f"{key}: {value_to_text(item)}"
            for key, item in value.items()
            if item not in (None, "", [], {})
        )
    if isinstance(value, (list, tuple, set)):
        return ", ".join(
            value_to_text(item) for item in value if item
        )
    return str(value).strip()


def json_safe(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {
                str(key): json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [json_safe(item) for item in value]
        return str(value)


def calculate_average(scores):
    valid = []

    for score in scores:
        try:
            if score is None:
                continue
            value = float(score)
            if 0 <= value <= 100:
                valid.append(value)
        except (TypeError, ValueError):
            continue

    return round(sum(valid) / len(valid), 2) if valid else 0.0


def optional_metric(result, name):
    value = result.get(name)

    if value is None or value == "":
        return None

    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Evaluator returned invalid metric: {name}"
        ) from exc

    if not 0 <= value <= 100:
        raise ValueError(
            f"{name} must be between 0 and 100."
        )

    return value


def validate_similarity(result):
    value = result.get("similarity", 0)

    if value is None or value == "":
        return 0.0

    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Evaluator returned invalid similarity."
        ) from exc

    if not 0 <= value <= 1:
        raise ValueError(
            "Similarity must be between 0 and 1."
        )

    return value


def error_response(message, status=400, **extra):
    return jsonify({
        "success": False,
        "error": message,
        **extra
    }), status


# ============================================================
# RESUME DATA STORAGE
# ============================================================

def get_resume_path(token):
    if not isinstance(token, str):
        return None

    if not re.fullmatch(r"[0-9a-f]{32}", token):
        return None

    return RESUME_DATA_DIR / f"{token}.json"


def load_resume_data(token):
    path = get_resume_path(token)

    if path is None or not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else None

    except (OSError, json.JSONDecodeError):
        logger.exception("Could not load resume data")
        return None


def save_resume_data(data):
    token = uuid.uuid4().hex
    path = get_resume_path(token)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)

    return token


# ============================================================
# RESUME / JOB DESCRIPTION HELPERS
# ============================================================

def guess_role(job_description, analysis):
    if job_description:
        for line in job_description.splitlines()[:5]:
            line = line.strip(" #:-\t")

            if len(line) <= 100 and re.search(
                r"\b(engineer|developer|analyst|scientist|intern|"
                r"manager|specialist|designer|consultant)\b",
                line,
                re.IGNORECASE
            ):
                return line

    summary = value_to_text(
        (analysis or {}).get("summary")
    )

    patterns = [
        (r"\bmachine learning\b|\bml engineer\b",
         "Machine Learning Engineer"),
        (r"\bdata scientist\b", "Data Scientist"),
        (r"\bdata analyst\b", "Data Analyst"),
        (r"\bsoftware engineer\b", "Software Engineer"),
        (r"\bbackend developer\b", "Backend Developer"),
        (r"\bfrontend developer\b", "Frontend Developer"),
        (r"\bpython developer\b", "Python Developer"),
        (r"\bai engineer\b|\bartificial intelligence\b",
         "AI Engineer"),
    ]

    for pattern, role in patterns:
        if re.search(pattern, summary, re.IGNORECASE):
            return role

    return ""


def safe_job_analysis(job_description):
    if not job_description or analyze_job_description is None:
        return {}

    try:
        result = analyze_job_description(job_description)
        return result if isinstance(result, dict) else {}
    except Exception:
        logger.exception("Job description analysis failed")
        return {}


def safe_resume_match(resume_text, job_description, resume_skills):
    if not job_description.strip():
        return {
            "match_score": None,
            "matched_skills": [],
            "missing_skills": [],
            "message": "Add a job description to calculate a match."
        }

    try:
        result = match_resume_to_job(
            resume_text,
            job_description,
            resume_skills=resume_skills
        )
    except TypeError:
        try:
            result = match_resume_to_job(
                resume_text, job_description
            )
        except Exception:
            logger.exception("Resume matching failed")
            return {
                "match_score": None,
                "matched_skills": [],
                "missing_skills": [],
                "message": "Matching could not be completed."
            }
    except Exception:
        logger.exception("Resume matching failed")
        return {
            "match_score": None,
            "matched_skills": [],
            "missing_skills": [],
            "message": "Matching could not be completed."
        }

    if not isinstance(result, dict):
        return {
            "match_score": None,
            "matched_skills": [],
            "missing_skills": [],
            "raw_result": json_safe(result)
        }

    result = dict(result)

    result.setdefault(
        "match_score",
        result.get(
            "match_percentage",
            result.get("score", result.get("overall_score"))
        )
    )
    result.setdefault(
        "matched_skills",
        result.get("matched", result.get("matchedSkills", []))
    )
    result.setdefault(
        "missing_skills",
        result.get("missing", result.get("missingSkills", []))
    )

    return json_safe(result)


# ============================================================
# RAG CONTEXT
# ============================================================

def get_rag_context(
    role,
    topic,
    skills="",
    job_description="",
    resume_token=None
):
    if not resume_token:
        return ""

    query = " ".join(
        part.strip()
        for part in [
            role,
            topic,
            skills,
            (job_description or "")[:500],
            "candidate skills experience projects interview questions"
        ]
        if isinstance(part, str) and part.strip()
    )

    try:
        result = retrieve_context(
            query,
            top_k=4,
            resume_token=resume_token
        )
        return str(result or "")[:2500]
    except Exception:
        logger.exception("RAG retrieval failed")
        return ""


# ============================================================
# INTERVIEW HELPERS
# ============================================================

def get_session_questions(session_id):
    return (
        InterviewQuestion.query
        .filter_by(session_id=session_id)
        .order_by(InterviewQuestion.id.asc())
        .all()
    )


def question_to_dict(question):
    created_at = getattr(question, "created_at", None)

    return {
        "id": question.id,
        "question": question.question or "",
        "expected_answer": question.expected_answer or "",
        "candidate_answer": question.candidate_answer or "",
        "topic": question.topic or "",
        "difficulty": question.difficulty or "",
        "score": question.score,
        "similarity": getattr(question, "similarity", None),
        "correctness": getattr(question, "correctness", None),
        "concept_coverage": getattr(
            question, "concept_coverage", None
        ),
        "completeness": getattr(question, "completeness", None),
        "relevance": getattr(question, "relevance", None),
        "verdict": question.verdict or "",
        "feedback": question.feedback or "",
        "improved_answer": getattr(
            question, "improved_answer", None
        ) or "",
        "evaluation_method": getattr(
            question, "evaluation_method", None
        ) or "",
        "created_at": (
            created_at.isoformat() if created_at else None
        )
    }


def interview_progress(interview):
    completed = InterviewQuestion.query.filter_by(
        session_id=interview.id
    ).count()

    target = TOTAL_INTERVIEW_QUESTIONS

    return {
        "target_questions": target,
        "completed_questions": completed,
        "remaining_questions": max(0, target - completed),
        "is_complete": completed >= target
    }


def get_interview_settings(session_id):
    settings = session.get("interview_settings", {})
    value = settings.get(str(session_id), {})
    return value if isinstance(value, dict) else {}


def save_interview_settings(session_id, settings):
    all_settings = session.get("interview_settings", {})

    if not isinstance(all_settings, dict):
        all_settings = {}

    all_settings[str(session_id)] = settings

    while len(all_settings) > 3:
        all_settings.pop(next(iter(all_settings)))

    session["interview_settings"] = all_settings
    session.modified = True


def build_topic_statistics(questions):
    topic_scores = {}

    for question in questions:
        try:
            if question.score is None:
                continue
            score = float(question.score)
        except (TypeError, ValueError):
            continue

        topic = question.topic or "General"
        topic_scores.setdefault(topic, []).append(score)

    averages = {
        topic: round(sum(scores) / len(scores), 2)
        for topic, scores in topic_scores.items()
        if scores
    }

    strong, weak, average, recommendations = [], [], [], []

    for topic, score in averages.items():
        item = {"topic": topic, "score": score}

        if score >= 80:
            strong.append(item)
            recommendations.append(
                f"{topic} is a strong area. Try advanced questions."
            )
        elif score < 60:
            weak.append(item)
            recommendations.append(
                f"Revise {topic} and practise more questions. "
                f"Average: {score}/100."
            )
        else:
            average.append(item)
            recommendations.append(
                f"Strengthen {topic} with practical examples. "
                f"Average: {score}/100."
            )

    strong.sort(key=lambda item: item["score"], reverse=True)
    weak.sort(key=lambda item: item["score"])

    if not recommendations:
        recommendations.append(
            "Complete interview questions to receive recommendations."
        )

    return {
        "topic_averages": averages,
        "strong_topics": strong,
        "weak_topics": weak,
        "average_topics": average,
        "recommendations": recommendations
    }


def normalized_interview_type(value):
    value = str(value or "Technical").strip().lower()

    mapping = {
        "technical": "Technical",
        "hr": "HR",
        "mixed": "Mixed"
    }

    return mapping.get(value)


# ============================================================
# PAGE ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/interview/setup")
def interview_setup():
    return render_template(
        "interview_setup.html",
        total_questions=TOTAL_INTERVIEW_QUESTIONS
    )


@app.route("/interview/<int:session_id>")
def interview_page(session_id):
    interview = db.get_or_404(InterviewSession, session_id)

    return render_template(
        "interview.html",
        interview=interview,
        total_questions=TOTAL_INTERVIEW_QUESTIONS
    )


@app.route("/interview/<int:session_id>/result")
def interview_result_page(session_id):
    interview = db.get_or_404(InterviewSession, session_id)

    return render_template(
        "interview_result.html",
        interview=interview,
        session_id=session_id
    )


# ============================================================
# RESUME ANALYSIS API
# ============================================================

@app.route("/api/resume/analyze", methods=["POST"])
def analyze_resume_api():
    uploaded = request.files.get("resume")

    if uploaded is None or not uploaded.filename:
        return error_response("Please select a resume file.")

    filename = secure_filename(uploaded.filename)

    if not filename:
        return error_response("Invalid filename.")

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_RESUME_EXTENSIONS:
        return error_response(
            "Only PDF, DOCX and TXT files are supported."
        )

    job_description = str(
        request.form.get("job_description", "")
    ).strip()[:15000]

    try:
        uploaded.stream.seek(0, os.SEEK_END)
        file_size = uploaded.stream.tell()
        uploaded.stream.seek(0)

        if file_size > app.config["MAX_CONTENT_LENGTH"]:
            return error_response(
                "Resume must be smaller than 8 MB.",
                413
            )

        parsed = parse_resume(uploaded, filename=filename)

        if not isinstance(parsed, dict):
            raise ValueError(
                "Resume parser must return a dictionary."
            )

        resume_text = str(parsed.get("text") or "").strip()

        if not resume_text:
            return error_response(
                "No readable text found in the resume. "
                "Scanned PDFs may require OCR."
            )

        analysis = analyze_resume(resume_text)

        if not isinstance(analysis, dict):
            analysis = {"summary": str(analysis)}

        resume_skills = as_list(analysis.get("skills"))
        jd_analysis = safe_job_analysis(job_description)
        job_skills = as_list(jd_analysis.get("skills"))

        match_result = safe_resume_match(
            resume_text,
            job_description,
            resume_skills
        )

        match_result.setdefault("resume_skills", resume_skills)
        match_result.setdefault("job_skills", job_skills)
        match_result.setdefault("matched_skills", [])
        match_result.setdefault("missing_skills", [])
        match_result.setdefault("match_score", None)

        suggested_role = guess_role(job_description, analysis)

        stored_data = {
            "filename": filename,
            "resume_text": resume_text[:30000],
            "analysis": json_safe(analysis),
            "jd_analysis": json_safe(jd_analysis),
            "match": json_safe(match_result),
            "job_description": job_description,
            "suggested_role": suggested_role
        }

        token = save_resume_data(stored_data)

        try:
            index_resume_text(
                resume_token=token,
                resume_text=resume_text
            )
        except Exception:
            path = get_resume_path(token)
            if path:
                path.unlink(missing_ok=True)

            logger.exception("Resume indexing failed")

            return error_response(
                "Resume parsed but indexing failed. "
                "Check the Flask terminal.",
                500
            )

        return jsonify({
            "success": True,
            "resume_token": token,
            "filename": filename,
            "word_count": len(resume_text.split()),
            "analysis": json_safe(analysis),
            "jd_analysis": json_safe(jd_analysis),
            "match": json_safe(match_result),
            "job_description": job_description,
            "suggested_role": suggested_role,
            "context": resume_text[:30000],
            "resume_text": resume_text[:30000]
        }), 200

    except ResumeParseError as exc:
        return error_response(str(exc), 400)
    except Exception:
        logger.exception("Resume analysis failed")
        return error_response(
            "Resume analysis failed. Check the Flask terminal.",
            500
        )


# ============================================================
# START INTERVIEW
# ============================================================

@app.route("/api/interview/start", methods=["POST"])
def start_interview():
    data = request.get_json(silent=True) or {}

    role = str(data.get("role") or "Software Engineer").strip()
    topic = str(data.get("topic") or "Machine Learning").strip()
    difficulty = str(data.get("difficulty") or "Medium").strip().title()
    interview_type = normalized_interview_type(
        data.get("interview_type")
    )

    skills = str(data.get("skills") or "").strip()
    job_description = str(
        data.get("job_description") or ""
    ).strip()
    resume_token = str(data.get("resume_token") or "").strip()

    if not role or not topic:
        return error_response("Role and topic are required.")

    if difficulty not in {"Easy", "Medium", "Hard"}:
        return error_response(
            "Difficulty must be Easy, Medium, or Hard."
        )

    if interview_type is None:
        return error_response(
            "Interview type must be Technical, HR, or Mixed."
        )

    if resume_token:
        resume_data = load_resume_data(resume_token)

        if resume_data is None:
            return error_response(
                "Resume token is invalid or expired. Upload again."
            )

        analysis = resume_data.get("analysis") or {}
        resume_skills = as_list(analysis.get("skills"))

        combined = [
            item.strip() for item in skills.split(",")
            if item.strip()
        ]
        combined.extend(
            value_to_text(item)
            for item in resume_skills
            if value_to_text(item)
        )
        skills = ", ".join(dict.fromkeys(combined))

        if not job_description:
            job_description = str(
                resume_data.get("job_description") or ""
            )

        if role == "Software Engineer":
            role = resume_data.get("suggested_role") or role

    try:
        interview = InterviewSession(
            role=role[:100],
            starting_topic=topic[:100],
            starting_difficulty=difficulty,
            total_questions=TOTAL_INTERVIEW_QUESTIONS
        )

        db.session.add(interview)
        db.session.commit()

        settings = {
            "role": role[:150],
            "topic": topic[:150],
            "difficulty": difficulty,
            "interview_type": interview_type,
            "skills": skills[:1000],
            "job_description": job_description[:3000],
            "resume_token": resume_token,
            "total_questions": TOTAL_INTERVIEW_QUESTIONS
        }

        save_interview_settings(interview.id, settings)
        session["interview_id"] = interview.id

        return jsonify({
            "success": True,
            "session_id": interview.id,
            "role": role,
            "topic": topic,
            "difficulty": difficulty,
            "interview_type": interview_type,
            "total_questions": TOTAL_INTERVIEW_QUESTIONS,
            "completed_questions": 0,
            "remaining_questions": TOTAL_INTERVIEW_QUESTIONS
        }), 201

    except SQLAlchemyError:
        db.session.rollback()
        logger.exception("Interview creation failed")
        return error_response(
            "Could not create interview session.",
            500
        )


# ============================================================
# GENERATE NEXT QUESTION
# ============================================================

@app.route(
    "/api/interview/<int:session_id>/question",
    methods=["POST"]
)
def generate_question_api(session_id):
    interview = db.get_or_404(InterviewSession, session_id)
    data = request.get_json(silent=True) or {}
    settings = get_interview_settings(session_id)
    progress = interview_progress(interview)

    if progress["is_complete"]:
        return error_response(
            "You have completed all interview questions.",
            409,
            interview_complete=True,
            **progress
        )

    role = str(
        settings.get("role") or interview.role
    ).strip()
    topic = str(
        data.get("topic") or settings.get("topic")
        or interview.starting_topic
    ).strip()
    difficulty = str(
        data.get("difficulty") or settings.get("difficulty")
        or interview.starting_difficulty
    ).strip().title()
    interview_type = normalized_interview_type(
        data.get("interview_type")
        or settings.get("interview_type")
    )
    skills = str(
        settings.get("skills") or ""
    ).strip()
    job_description = str(
        settings.get("job_description") or ""
    ).strip()

    if difficulty not in {"Easy", "Medium", "Hard"}:
        return error_response("Invalid difficulty.")

    if interview_type is None:
        return error_response("Invalid interview type.")

    previous_questions = data.get("previous_questions", [])
    if not isinstance(previous_questions, list):
        previous_questions = []

    previous = []
    seen = set()

    for item in previous_questions:
        if isinstance(item, str) and item.strip():
            cleaned = item.strip()
            if cleaned.casefold() not in seen:
                previous.append(cleaned)
                seen.add(cleaned.casefold())

    try:
        saved_questions = get_session_questions(session_id)

        for item in saved_questions:
            if item.question:
                key = item.question.strip().casefold()
                if key not in seen:
                    previous.append(item.question)
                    seen.add(key)

        context_parts = []
        resume_token = settings.get("resume_token", "")

        rag_context = get_rag_context(
            role=role,
            topic=topic,
            skills=skills,
            job_description=job_description,
            resume_token=resume_token
        )

        if rag_context:
            context_parts.append(
                "RETRIEVED RESUME CONTEXT:\n" + rag_context
            )

        resume_data = load_resume_data(resume_token)

        if resume_data:
            analysis = resume_data.get("analysis") or {}

            resume_context = (
                f"RESUME SUMMARY: {value_to_text(analysis.get('summary'))}\n"
                f"RESUME SKILLS: {value_to_text(analysis.get('skills'))}\n"
                f"RESUME PROJECTS: {value_to_text(analysis.get('projects'))}\n"
                f"RESUME EDUCATION: {value_to_text(analysis.get('education'))}\n"
                f"RESUME EXPERIENCE: {value_to_text(analysis.get('experience'))}\n"
                f"JOB DESCRIPTION: {job_description[:1500]}\n"
                f"RESUME EXCERPT: {resume_data.get('resume_text', '')[:2000]}"
            )

            context_parts.append(
                "CANDIDATE RESUME:\n" + resume_context
            )

            combined = [
                item.strip() for item in skills.split(",")
                if item.strip()
            ]
            combined.extend(
                value_to_text(item)
                for item in as_list(analysis.get("skills"))
                if value_to_text(item)
            )
            skills = ", ".join(dict.fromkeys(combined))

        context = "\n\n".join(context_parts)[:6000]

        content = generate_interview_content(
            role=role,
            skill=topic,
            difficulty=difficulty,
            context=context,
            previous_questions=previous,
            interview_type=interview_type,
            skills=skills,
            job_description=job_description
        )

        if not isinstance(content, dict):
            raise ValueError(
                "Question generator must return a dictionary."
            )

        question_text = str(content.get("question") or "").strip()
        expected_answer = str(
            content.get("expected_answer") or ""
        ).strip()

        if not question_text or not expected_answer:
            raise ValueError(
                "Generated question or expected answer is empty."
            )

        # A question is returned to the browser but is saved only
        # when its answer is evaluated. Do not increment progress here.
        return jsonify({
            "success": True,
            "question": question_text,
            "expected_answer": expected_answer,
            "topic": content.get("topic") or topic,
            "difficulty": content.get("difficulty") or difficulty,
            "interview_type": interview_type,
            "source": "configured_question_generator",
            **progress
        }), 200

    except Exception:
        logger.exception("Question generation failed")
        return error_response(
            "Question generation failed. Check the Flask terminal.",
            500
        )


# ============================================================
# EVALUATE AND SAVE ANSWER
# ============================================================

@app.route(
    "/api/interview/<int:session_id>/answer",
    methods=["POST"]
)
def evaluate_interview_answer(session_id):
    interview = db.get_or_404(InterviewSession, session_id)
    data = request.get_json(silent=True) or {}

    completed = InterviewQuestion.query.filter_by(
        session_id=session_id
    ).count()

    if completed >= TOTAL_INTERVIEW_QUESTIONS:
        return error_response(
            "Interview question limit reached.",
            409,
            saved=False,
            interview_complete=True,
            target_questions=TOTAL_INTERVIEW_QUESTIONS,
            completed_questions=completed,
            remaining_questions=0
        )

    question_text = str(data.get("question") or "").strip()
    expected_answer = str(
        data.get("expected_answer") or ""
    ).strip()
    candidate_answer = str(
        data.get("candidate_answer") or ""
    ).strip()

    topic = str(
        data.get("topic") or interview.starting_topic
    ).strip()
    difficulty = str(
        data.get("difficulty") or interview.starting_difficulty
    ).strip().title()

    if not question_text:
        return error_response("Question is missing.")

    if not expected_answer:
        return error_response(
            "Expected answer is missing. Generate the question again."
        )

    if not candidate_answer:
        return error_response(
            "Please enter your answer before submitting."
        )

    if difficulty not in {"Easy", "Medium", "Hard"}:
        difficulty = interview.starting_difficulty

    try:
        result = evaluate_answer(
            question=question_text,
            reference_answer=expected_answer,
            candidate_answer=candidate_answer
        )

        if not isinstance(result, dict):
            raise ValueError(
                "Evaluator did not return a dictionary."
            )

        if result.get("score") is None:
            raise ValueError(
                "Evaluator did not return an overall score."
            )

        score = float(result["score"])

        if not 0 <= score <= 100:
            raise ValueError(
                "Overall score must be between 0 and 100."
            )

        similarity = validate_similarity(result)

        saved = InterviewQuestion(
            session_id=session_id,
            question=question_text,
            expected_answer=expected_answer,
            candidate_answer=candidate_answer,
            topic=topic[:100],
            difficulty=difficulty[:20],
            score=score,
            similarity=similarity,
            correctness=optional_metric(result, "correctness"),
            concept_coverage=optional_metric(
                result, "concept_coverage"
            ),
            completeness=optional_metric(result, "completeness"),
            relevance=optional_metric(result, "relevance"),
            verdict=str(
                result.get("verdict") or "Needs Improvement"
            )[:50],
            feedback=str(result.get("feedback") or ""),
            improved_answer=str(
                result.get("improved_answer") or ""
            ),
            evaluation_method=str(
                result.get("evaluation_method") or "unknown"
            )[:50]
        )

        db.session.add(saved)
        db.session.commit()

        response = question_to_dict(saved)
        response["success"] = True
        response["saved"] = True
        response.update(interview_progress(interview))

        return jsonify(response), 200

    except SQLAlchemyError:
        db.session.rollback()
        logger.exception("Saving answer failed")
        return error_response(
            "Could not save the evaluation. Check database schema and logs.",
            500,
            saved=False
        )

    except Exception:
        db.session.rollback()
        logger.exception("Answer evaluation failed")
        return error_response(
            "Answer evaluation failed. Check the Flask terminal.",
            500,
            saved=False
        )


# ============================================================
# INTERVIEW RESULT API
# ============================================================

@app.route(
    "/api/interview/<int:session_id>/result",
    methods=["GET"]
)
def get_interview_result(session_id):
    interview = db.get_or_404(InterviewSession, session_id)
    questions = get_session_questions(session_id)

    stats = build_topic_statistics(questions)
    progress = interview_progress(interview)

    scores = [
        question.score
        for question in questions
        if question.score is not None
    ]

    return jsonify({
        "success": True,
        "session_id": session_id,
        "role": interview.role,
        "topic": interview.starting_topic,
        "difficulty": interview.starting_difficulty,
        "total_questions": TOTAL_INTERVIEW_QUESTIONS,
        "answered_questions": len(questions),
        **progress,
        "average_score": calculate_average(scores),
        "strong_topics": stats["strong_topics"],
        "weak_topics": stats["weak_topics"],
        "average_topics": stats["average_topics"],
        "topic_averages": stats["topic_averages"],
        "recommendations": stats["recommendations"],
        "questions": [
            question_to_dict(question)
            for question in questions
        ]
    }), 200


# ============================================================
# INTERVIEW HISTORY
# ============================================================

@app.route("/api/interviews", methods=["GET"])
def get_interviews():
    interviews = (
        InterviewSession.query
        .order_by(InterviewSession.created_at.desc())
        .all()
    )

    results = []

    for interview in interviews:
        questions = get_session_questions(interview.id)
        scores = [
            q.score for q in questions if q.score is not None
        ]

        completed = len(questions)
        target = TOTAL_INTERVIEW_QUESTIONS

        results.append({
            "id": interview.id,
            "role": interview.role,
            "topic": interview.starting_topic,
            "difficulty": interview.starting_difficulty,
            "average_score": calculate_average(scores),
            "total_questions": target,
            "answered_questions": completed,
            "target_questions": target,
            "completed_questions": completed,
            "remaining_questions": max(0, target - completed),
            "created_at": (
                interview.created_at.isoformat()
                if interview.created_at else None
            )
        })

    return jsonify({
        "success": True,
        "interviews": results
    }), 200


@app.route(
    "/api/interviews/<int:session_id>",
    methods=["GET"]
)
def get_single_interview(session_id):
    interview = db.get_or_404(InterviewSession, session_id)
    questions = get_session_questions(session_id)

    return jsonify({
        "success": True,
        "id": interview.id,
        "role": interview.role,
        "topic": interview.starting_topic,
        "difficulty": interview.starting_difficulty,
        "total_questions": TOTAL_INTERVIEW_QUESTIONS,
        "answered_questions": len(questions),
        **interview_progress(interview),
        "questions": [
            question_to_dict(question) for question in questions
        ]
    }), 200


@app.route(
    "/api/interviews/<int:session_id>",
    methods=["DELETE"]
)
def delete_interview(session_id):
    interview = db.get_or_404(InterviewSession, session_id)

    try:
        db.session.delete(interview)
        db.session.commit()

        settings = session.get("interview_settings", {})
        if isinstance(settings, dict):
            settings.pop(str(session_id), None)

        session["interview_settings"] = settings
        session.modified = True

        if session.get("interview_id") == session_id:
            session.pop("interview_id", None)

        return jsonify({
            "success": True,
            "deleted": True
        }), 200

    except SQLAlchemyError:
        db.session.rollback()
        logger.exception("Interview deletion failed")
        return error_response(
            "Could not delete interview.",
            500
        )


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

@app.route("/api/statistics", methods=["GET"])
def get_statistics():
    interviews = InterviewSession.query.all()
    questions = InterviewQuestion.query.all()

    scores = [
        question.score
        for question in questions
        if question.score is not None
    ]

    stats = build_topic_statistics(questions)

    return jsonify({
        "success": True,
        "total_interviews": len(interviews),
        "total_questions": len(questions),
        "average_score": calculate_average(scores),
        "topic_averages": stats["topic_averages"],
        "strong_topics": stats["strong_topics"],
        "weak_topics": stats["weak_topics"],
        "average_topics": stats["average_topics"],
        "recommendations": stats["recommendations"]
    }), 200


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def file_too_large(error):
    if request.path.startswith("/api/"):
        return error_response(
            "Uploaded file is too large. Maximum size is 8 MB.",
            413
        )

    return "File too large (maximum size 8 MB).", 413


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return error_response(
            "Requested interview or resource was not found.",
            404
        )

    return error


@app.errorhandler(500)
def internal_server_error(error):
    if request.path.startswith("/api/"):
        return error_response(
            "An internal server error occurred.",
            500
        )

    return error


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
