def calculate_topic_averages(topic_scores):

    averages = {}

    for topic, scores in topic_scores.items():

        if not scores:
            continue

        average = sum(scores) / len(scores)

        averages[topic] = round(average, 2)

    return averages


def analyze_topics(topic_scores):

    averages = calculate_topic_averages(topic_scores)

    strong_topics = []
    weak_topics = []
    average_topics = []

    for topic, average in averages.items():

        if average >= 8:
            strong_topics.append({
                "topic": topic,
                "score": average
            })

        elif average < 6:
            weak_topics.append({
                "topic": topic,
                "score": average
            })

        else:
            average_topics.append({
                "topic": topic,
                "score": average
            })

    # Highest score first
    strong_topics.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Lowest score first
    weak_topics.sort(
        key=lambda x: x["score"]
    )

    return {
        "averages": averages,
        "strong_topics": strong_topics,
        "weak_topics": weak_topics,
        "average_topics": average_topics
    }


def generate_recommendations(analysis):

    recommendations = []

    for item in analysis["weak_topics"]:

        topic = item["topic"]
        score = item["score"]

        recommendations.append(
            f"Practice {topic} more. "
            f"Current average score is {score}/10."
        )

    for item in analysis["strong_topics"]:

        topic = item["topic"]

        recommendations.append(
            f"{topic} is a strong area. "
            f"Try more advanced questions."
        )

    if not recommendations:

        recommendations.append(
            "Keep practicing different topics "
            "to build a balanced skill profile."
        )

    return recommendations


def get_topic_analysis(topic_scores):

    analysis = analyze_topics(topic_scores)

    recommendations = generate_recommendations(
        analysis
    )

    analysis["recommendations"] = recommendations

    return analysis


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    topic_scores = {

        "Python": [8, 9, 8],

        "Machine Learning": [7, 8, 7],

        "SQL": [4, 5, 4],

        "Pandas": [5, 6, 5]

    }


    result = get_topic_analysis(
        topic_scores
    )


    print("\n========================================")
    print("       TOPIC PERFORMANCE ANALYSIS")
    print("========================================")


    print("\nTopic Averages:")

    for topic, score in result["averages"].items():

        print(
            f"{topic}: {score}/10"
        )


    print("\nStrong Topics:")

    for item in result["strong_topics"]:

        print(
            f"✓ {item['topic']}: "
            f"{item['score']}/10"
        )


    print("\nWeak Topics:")

    for item in result["weak_topics"]:

        print(
            f"⚠ {item['topic']}: "
            f"{item['score']}/10"
        )


    print("\nRecommendations:")

    for recommendation in result["recommendations"]:

        print(
            "→",
            recommendation
        )