from resume.resume_parser import parse_resume
from resume.skill_extractor import extract_skills
from resume.jd_analyzer import extract_jd_skills
from resume.semantic_matcher import semantic_skill_matching

from rag.retriever import retrieve
from interview.question_generator import generate_interview_content


# =========================================================
# CREATE INTERVIEW PLAN
# =========================================================

def create_interview_plan(
    matched_skills,
    missing_skills,
    total_questions=10
):

    topics = []

    # Matched skills get higher priority
    for item in matched_skills:

        topics.append({
            "topic": item["jd_skill"],
            "source": "Resume + JD",
            "priority": "High",
            "similarity": item["similarity"]
        })

    # Missing skills
    for item in missing_skills:

        topics.append({
            "topic": item["skill"],
            "source": "JD Requirement",
            "priority": "Medium",
            "similarity": item["best_similarity"]
        })

    if not topics:
        return []

    # -----------------------------------------------------
    # Give more importance to matched skills
    # -----------------------------------------------------

    weighted_topics = []

    for item in topics:

        if item["priority"] == "High":
            weight = 2
        else:
            weight = 1

        for _ in range(weight):
            weighted_topics.append(item)

    # -----------------------------------------------------
    # Allocate questions
    # -----------------------------------------------------

    plan = []

    total_weight = len(weighted_topics)

    question_counts = {}

    for item in weighted_topics:

        topic = item["topic"]

        if topic not in question_counts:
            question_counts[topic] = 0

    for topic in question_counts:

        count = sum(
            1
            for item in weighted_topics
            if item["topic"] == topic
        )

        question_counts[topic] = count

    total_allocated = sum(
        question_counts.values()
    )

    # Normalize to requested total
    if total_allocated > total_questions:

        while total_allocated > total_questions:

            largest_topic = max(
                question_counts,
                key=question_counts.get
            )

            if question_counts[largest_topic] > 1:

                question_counts[largest_topic] -= 1
                total_allocated -= 1

            else:
                break

    elif total_allocated < total_questions:

        topics_list = list(
            question_counts.keys()
        )

        index = 0

        while total_allocated < total_questions:

            topic = topics_list[
                index % len(topics_list)
            ]

            question_counts[topic] += 1

            total_allocated += 1

            index += 1

    # -----------------------------------------------------
    # Build final plan
    # -----------------------------------------------------

    for item in topics:

        topic = item["topic"]

        # Avoid duplicate topic entries
        if any(
            x["topic"] == topic
            for x in plan
        ):
            continue

        plan.append({

            "topic": topic,

            "source":
                item["source"],

            "priority":
                item["priority"],

            "similarity":
                item["similarity"],

            "question_count":
                question_counts[topic]

        })

    return plan


# =========================================================
# GET RAG CONTEXT
# =========================================================

def get_topic_context(
    role,
    topic
):

    query = f"{role} {topic}"

    results = retrieve(
        query=query,
        top_k=2
    )

    context_parts = []

    for result in results:

        context_parts.append(
            result["chunk"]
        )

    if not context_parts:

        return "No relevant context found."

    return "\n".join(
        context_parts
    )


# =========================================================
# GENERATE PERSONALIZED QUESTIONS
# =========================================================

def generate_personalized_interview(
    role,
    plan,
    difficulty="Medium"
):

    interview_questions = []

    for item in plan:

        topic = item["topic"]

        question_count = item[
            "question_count"
        ]

        # ---------------------------------------------
        # RAG
        # ---------------------------------------------

        context = get_topic_context(
            role=role,
            topic=topic
        )

        # ---------------------------------------------
        # Generate questions
        # ---------------------------------------------

        for _ in range(question_count):

            result = generate_interview_content(

                role=role,

                skill=topic,

                difficulty=difficulty,

                context=context

            )

            interview_questions.append({

                "question":
                    result["question"],

                "expected_answer":
                    result["expected_answer"],

                "topic":
                    topic,

                "difficulty":
                    difficulty,

                "source":
                    item["source"],

                "priority":
                    item["priority"],

                "similarity":
                    item["similarity"]

            })

    return interview_questions


# =========================================================
# COMPLETE PERSONALIZED PIPELINE
# =========================================================

def build_personalized_interview(
    resume_path,
    jd_text,
    role,
    difficulty="Medium",
    total_questions=10
):

    print("\n")
    print("=" * 60)
    print("AI PERSONALIZED INTERVIEW")
    print("=" * 60)

    # =====================================================
    # 1. RESUME
    # =====================================================

    print("\n[1] Reading resume...")

    resume_text = parse_resume(
        resume_path
    )

    print(
        "Resume characters:",
        len(resume_text)
    )

    # =====================================================
    # 2. RESUME SKILLS
    # =====================================================

    print("\n[2] Extracting resume skills...")

    resume_skills = extract_skills(
        resume_text
    )

    print(
        "Resume skills:",
        resume_skills
    )

    # =====================================================
    # 3. JD SKILLS
    # =====================================================

    print("\n[3] Extracting JD skills...")

    jd_skills = extract_jd_skills(
        jd_text
    )

    print(
        "JD skills:",
        jd_skills
    )

    # =====================================================
    # 4. SEMANTIC MATCHING
    # =====================================================

    print("\n[4] Semantic skill matching...")

    matching = semantic_skill_matching(

        resume_skills=
            resume_skills,

        jd_skills=
            jd_skills,

        threshold=0.65

    )

    # =====================================================
    # 5. DISPLAY MATCHING
    # =====================================================

    print("\nMatched Skills:")

    for item in matching["matched"]:

        print(
            f"✓ {item['jd_skill']} "
            f"← {item['resume_skill']} "
            f"({item['similarity']})"
        )

    print("\nMissing Skills:")

    for item in matching["missing"]:

        print(
            f"⚠ {item['skill']} "
            f"(similarity: "
            f"{item['best_similarity']})"
        )

    print(
        "\nMatch Percentage:",
        matching["match_percentage"],
        "%"
    )

    # =====================================================
    # 6. INTERVIEW PLAN
    # =====================================================

    print("\n[5] Creating interview plan...")

    plan = create_interview_plan(

        matched_skills=
            matching["matched"],

        missing_skills=
            matching["missing"],

        total_questions=
            total_questions

    )

    print("\nInterview Plan:")

    for item in plan:

        print(
            f"✓ {item['topic']} "
            f"→ {item['question_count']} "
            f"questions "
            f"[{item['priority']}]"
        )

    # =====================================================
    # 7. GENERATE QUESTIONS
    # =====================================================

    print(
        "\n[6] Generating personalized questions..."
    )

    questions = generate_personalized_interview(

        role=role,

        plan=plan,

        difficulty=difficulty

    )

    # =====================================================
    # 8. RETURN EVERYTHING
    # =====================================================

    return {

        "resume_skills":
            resume_skills,

        "jd_skills":
            jd_skills,

        "matched_skills":
            matching["matched"],

        "missing_skills":
            matching["missing"],

        "match_percentage":
            matching["match_percentage"],

        "interview_plan":
            plan,

        "questions":
            questions

    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    resume_path = (
        "uploads/resumes/Resume.pdf"
    )

    jd_text = """
    We are looking for a Machine Learning Engineer.

    Required skills:

    Python
    SQL
    Machine Learning
    Pandas
    Scikit-learn
    Natural Language Processing
    Docker
    AWS
    PyTorch
    """

    result = build_personalized_interview(

        resume_path=resume_path,

        jd_text=jd_text,

        role="Machine Learning Engineer",

        difficulty="Medium",

        total_questions=5

    )

    # =====================================================
    # FINAL QUESTIONS
    # =====================================================

    print("\n")
    print("=" * 60)
    print("PERSONALIZED INTERVIEW QUESTIONS")
    print("=" * 60)

    for i, question in enumerate(
        result["questions"],
        start=1
    ):

        print("\n")

        print(
            f"Question {i}"
        )

        print(
            "-" * 40
        )

        print(
            "Topic:",
            question["topic"]
        )

        print(
            "Difficulty:",
            question["difficulty"]
        )

        print(
            "Priority:",
            question["priority"]
        )

        print(
            "Source:",
            question["source"]
        )

        print(
            "Similarity:",
            question["similarity"]
        )

        print(
            "\nQuestion:",
            question["question"]
        )

        print(
            "\nExpected Answer:",
            question["expected_answer"]
        )