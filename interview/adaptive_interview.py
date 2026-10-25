from interview.question_generator import generate_interview_content
from interview.evaluator import evaluate_answer
from interview.adaptive_engine import adaptive_decision
from interview.topic_analyzer import get_topic_analysis


# =========================================================
# GET RAG CONTEXT
# =========================================================

def get_context(retrieve, role, topic):

    results = retrieve(
        query=f"{role} {topic}",
        top_k=2
    )

    context = []

    for result in results:
        context.append(
            result["chunk"]
        )

    if not context:
        return "No relevant context available."

    return "\n".join(context)


# =========================================================
# RUN ADAPTIVE INTERVIEW
# =========================================================

def run_adaptive_interview(
    role,
    topics,
    starting_difficulty="Medium",
    total_questions=5,
    retrieve_function=None
):

    topic_scores = {}

    current_topic = topics[0]
    current_difficulty = starting_difficulty

    results = []

    print("\n")
    print("=" * 60)
    print("        ADAPTIVE AI INTERVIEW")
    print("=" * 60)

    for question_number in range(
        1,
        total_questions + 1
    ):

        print("\n")
        print("=" * 60)

        print(
            f"QUESTION {question_number}/{total_questions}"
        )

        print("=" * 60)

        print(
            "Topic:",
            current_topic
        )

        print(
            "Difficulty:",
            current_difficulty
        )

        # -------------------------------------------------
        # RAG CONTEXT
        # -------------------------------------------------

        context = ""

        if retrieve_function:

            context = get_context(
                retrieve_function,
                role,
                current_topic
            )

        # -------------------------------------------------
        # GENERATE QUESTION
        # -------------------------------------------------

        content = generate_interview_content(

            role=role,

            skill=current_topic,

            difficulty=current_difficulty,

            context=context

        )

        question = content[
            "question"
        ]

        expected_answer = content[
            "expected_answer"
        ]

        print("\nQuestion:")
        print(question)

        # -------------------------------------------------
        # CANDIDATE ANSWER
        # -------------------------------------------------

        answer = input(
            "\nYour Answer: "
        ).strip()

        # -------------------------------------------------
        # EVALUATION
        # -------------------------------------------------

        evaluation = evaluate_answer(

            question=question,

            reference_answer=expected_answer,

            candidate_answer=answer

        )

        score = evaluation[
            "score"
        ]

        print("\n")
        print("-" * 60)

        print(
            f"Score      : {score}/10"
        )

        print(
            f"Similarity : "
            f"{evaluation['similarity']:.2f}"
        )

        print(
            f"Verdict    : "
            f"{evaluation['verdict']}"
        )

        print(
            f"Feedback   : "
            f"{evaluation['feedback']}"
        )

        print("-" * 60)

        # -------------------------------------------------
        # STORE TOPIC SCORE
        # -------------------------------------------------

        if current_topic not in topic_scores:

            topic_scores[
                current_topic
            ] = []

        topic_scores[
            current_topic
        ].append(score)

        # -------------------------------------------------
        # STORE QUESTION RESULT
        # -------------------------------------------------

        results.append({

            "question_number":
                question_number,

            "question":
                question,

            "answer":
                answer,

            "expected_answer":
                expected_answer,

            "topic":
                current_topic,

            "difficulty":
                current_difficulty,

            "score":
                score,

            "verdict":
                evaluation["verdict"],

            "feedback":
                evaluation["feedback"]

        })

        # -------------------------------------------------
        # ADAPTIVE DECISION
        # -------------------------------------------------

        decision = adaptive_decision(

            score=score,

            current_difficulty=
                current_difficulty,

            topic_scores=
                topic_scores,

            current_topic=
                current_topic

        )

        next_topic = decision[
            "next_topic"
        ]

        next_difficulty = decision[
            "next_difficulty"
        ]

        print("\n")
        print("ADAPTIVE DECISION")

        print(
            "Next Topic:",
            next_topic
        )

        print(
            "Next Difficulty:",
            next_difficulty
        )

        # -------------------------------------------------
        # UPDATE
        # -------------------------------------------------

        current_topic = next_topic

        current_difficulty = (
            next_difficulty
        )

    # =====================================================
    # FINAL ANALYSIS
    # =====================================================

    analysis = get_topic_analysis(
        topic_scores
    )

    print("\n")
    print("=" * 60)
    print("         INTERVIEW COMPLETED")
    print("=" * 60)

    print("\nTopic Performance:")

    for topic, score_list in topic_scores.items():

        average = (
            sum(score_list)
            / len(score_list)
        )

        print(
            f"{topic}: "
            f"{average:.2f}/10"
        )

    print("\nStrong Topics:")

    if analysis["strong_topics"]:

        for item in analysis[
            "strong_topics"
        ]:

            print(
                f"✓ {item['topic']} "
                f"→ {item['score']}/10"
            )

    else:

        print(
            "No strong topics detected."
        )

    print("\nWeak Topics:")

    if analysis["weak_topics"]:

        for item in analysis[
            "weak_topics"
        ]:

            print(
                f"⚠ {item['topic']} "
                f"→ {item['score']}/10"
            )

    else:

        print(
            "No weak topics detected."
        )

    print("\nRecommendations:")

    for recommendation in analysis[
        "recommendations"
    ]:

        print(
            "→",
            recommendation
        )

    return {

        "questions":
            results,

        "topic_scores":
            topic_scores,

        "analysis":
            analysis

    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    from rag.retriever import retrieve

    topics = [

        "Python",

        "Machine Learning",

        "SQL",

        "Pandas"

    ]

    result = run_adaptive_interview(

        role="Machine Learning Engineer",

        topics=topics,

        starting_difficulty="Medium",

        total_questions=5,

        retrieve_function=retrieve

    )