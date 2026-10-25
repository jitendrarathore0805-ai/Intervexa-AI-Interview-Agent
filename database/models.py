
from datetime import datetime

from database.db import db


class InterviewSession(db.Model):
    __tablename__ = "interview_session"

    id = db.Column(db.Integer, primary_key=True)

    role = db.Column(
        db.String(100),
        nullable=False
    )

    starting_topic = db.Column(
        db.String(100),
        nullable=False
    )

    starting_difficulty = db.Column(
        db.String(20),
        nullable=False
    )

    total_questions = db.Column(
        db.Integer,
        nullable=False,
        default=10
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow
    )

    questions = db.relationship(
        "InterviewQuestion",
        backref="session",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="InterviewQuestion.id"
    )

    def __repr__(self):
        return (
            f"<InterviewSession "
            f"id={self.id}, role={self.role!r}>"
        )


class InterviewQuestion(db.Model):
    __tablename__ = "interview_question"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    session_id = db.Column(
        db.Integer,
        db.ForeignKey("interview_session.id"),
        nullable=False,
        index=True
    )

    question = db.Column(
        db.Text,
        nullable=False
    )

    expected_answer = db.Column(
        db.Text,
        nullable=False
    )

    candidate_answer = db.Column(
        db.Text,
        nullable=True
    )

    topic = db.Column(
        db.String(100),
        nullable=False
    )

    difficulty = db.Column(
        db.String(20),
        nullable=False
    )

    # Overall evaluation score: 0–100
    score = db.Column(
        db.Float,
        nullable=True,
        default=0.0
    )

    # Semantic similarity: 0–1
    similarity = db.Column(
        db.Float,
        nullable=True,
        default=0.0
    )

    # Detailed evaluation metrics: 0–100
    correctness = db.Column(
        db.Float,
        nullable=True
    )

    concept_coverage = db.Column(
        db.Float,
        nullable=True
    )

    completeness = db.Column(
        db.Float,
        nullable=True
    )

    relevance = db.Column(
        db.Float,
        nullable=True
    )

    verdict = db.Column(
        db.String(50),
        nullable=True
    )

    feedback = db.Column(
        db.Text,
        nullable=True
    )

    improved_answer = db.Column(
        db.Text,
        nullable=True
    )

    evaluation_method = db.Column(
        db.String(100),
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow
    )

    def __repr__(self):
        return (
            f"<InterviewQuestion "
            f"id={self.id}, "
            f"session_id={self.session_id}, "
            f"score={self.score}>"
        )
