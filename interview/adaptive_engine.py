def get_next_difficulty(score, current_difficulty):
    """
    Decide next question difficulty based on score.
    """

    if score >= 8:
        # Candidate performed well
        if current_difficulty == "Easy":
            return "Medium"

        elif current_difficulty == "Medium":
            return "Hard"

        else:
            return "Hard"

    elif score >= 5:
        # Average performance
        return current_difficulty

    else:
        # Weak performance
        if current_difficulty == "Hard":
            return "Medium"

        elif current_difficulty == "Medium":
            return "Easy"

        else:
            return "Easy"


def choose_next_topic(topic_scores, current_topic):
    """
    Select the next topic.

    topic_scores example:

    {
        "Python": [8, 9],
        "Pandas": [5, 6],
        "Machine Learning": [4, 5]
    }
    """

    if not topic_scores:
        return current_topic

    # Calculate average score of each topic
    averages = {}

    for topic, scores in topic_scores.items():

        if scores:
            averages[topic] = sum(scores) / len(scores)

    # Find weakest topic
    weakest_topic = min(
        averages,
        key=averages.get
    )

    # If current topic is weak,
    # continue with it for a follow-up question
    if averages[current_topic] < 6:
        return current_topic

    return weakest_topic


def adaptive_decision(
    score,
    current_difficulty,
    topic_scores,
    current_topic
):

    next_difficulty = get_next_difficulty(
        score,
        current_difficulty
    )

    next_topic = choose_next_topic(
        topic_scores,
        current_topic
    )

    return {
        "next_difficulty": next_difficulty,
        "next_topic": next_topic
    }


if __name__ == "__main__":

    print("========== ADAPTIVE ENGINE TEST ==========")

    topic_scores = {
        "Python": [8, 9],
        "Pandas": [6, 7],
        "Machine Learning": [4, 5],
        "SQL": [8, 8]
    }

    current_topic = "Machine Learning"
    current_difficulty = "Medium"
    current_score = 5

    result = adaptive_decision(
        score=current_score,
        current_difficulty=current_difficulty,
        topic_scores=topic_scores,
        current_topic=current_topic
    )

    print("\nCurrent Topic:", current_topic)
    print("Current Difficulty:", current_difficulty)
    print("Current Score:", current_score)

    print("\nNext Topic:", result["next_topic"])
    print("Next Difficulty:", result["next_difficulty"])

    print("\n==========================================")