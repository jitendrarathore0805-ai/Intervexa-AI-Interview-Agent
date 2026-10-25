import logging

logging.basicConfig(level=logging.INFO)



from interview.question_generator import (
    MODEL_NAME,
    generate_question,
    generate_expected_answer,
)

print("Model:", MODEL_NAME)

try:
    question = generate_question(
        role="Machine Learning Engineer",
        skill="CNN",
        difficulty="medium",
        interview_type="technical",
        skills=["Python", "Machine Learning", "CNN"],
    )

    print("\nQuestion:\n", question)

    answer = generate_expected_answer(
        question=question,
        role="Machine Learning Engineer",
        skill="CNN",
        difficulty="medium",
        interview_type="technical",
    )

    print("\nExpected answer:\n", answer)

except Exception as exc:
    print("\nERROR:", exc)