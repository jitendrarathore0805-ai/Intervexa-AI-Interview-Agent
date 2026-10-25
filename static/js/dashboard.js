
"use strict";

const $ = (selector) => document.querySelector(selector);

/* =====================================================
   FORMAT HELPERS
===================================================== */

function formatScore(value) {
    const score = Number(value);

    if (!Number.isFinite(score)) {
        return "Not evaluated";
    }

    return `${Math.max(0, Math.min(100, score)).toFixed(1)}/100`;
}

function formatDate(value) {
    if (!value) return "Date unavailable";

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return "Date unavailable";
    }

    return date.toLocaleDateString("en-IN", {
        day: "2-digit",
        month: "short",
        year: "numeric"
    });
}

function makeElement(tag, className, text) {
    const element = document.createElement(tag);

    if (className) {
        element.className = className;
    }

    if (text !== undefined && text !== null) {
        element.textContent = String(text);
    }

    return element;
}

/* =====================================================
   API HELPER
   Existing API behaviour preserved
===================================================== */

async function fetchJSON(url, options = {}) {
    const response = await fetch(url, {
        ...options,
        headers: {
            ...(options.headers || {}),
            "Content-Type": "application/json"
        }
    });

    const data = await response.json();

    if (!response.ok || data.success === false) {
        throw new Error(data.error || "Request failed.");
    }

    return data;
}

/* =====================================================
   INJECT DASHBOARD HISTORY STYLES

   Isse Topicpython jaise concatenated text ki jagah
   labels aur values separate rows mein dikhenge.
===================================================== */

function addDashboardStyles() {
    if (document.getElementById("intervexa-dashboard-styles")) {
        return;
    }

    const style = document.createElement("style");
    style.id = "intervexa-dashboard-styles";

    style.textContent = `
        #interviewList {
            width: 100%;
            max-width: 100%;
            min-width: 0;
        }

        #interviewList .history-card {
            display: block;
            width: 100%;
            max-width: 100%;
            min-width: 0;
            margin: 14px 0;
            padding: 22px;
            overflow: hidden;
            overflow-wrap: anywhere;
            color: #f3f6ff;
            background:
                linear-gradient(135deg, #121c2e, #0b1220);
            border: 1px solid #2a3a57;
            border-radius: 16px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.16);
            transition:
                border-color 0.2s ease,
                transform 0.2s ease,
                box-shadow 0.2s ease;
        }

        #interviewList .history-card:hover {
            border-color: #4265a0;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.22);
            transform: translateY(-2px);
        }

        #interviewList .history-heading {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 9px 16px;
            margin-bottom: 18px;
            padding-bottom: 15px;
            border-bottom: 1px solid #25334b;
        }

        #interviewList .history-role {
            flex: 1 1 180px;
            min-width: 0;
            margin: 0;
            color: #f4f7ff;
            font-size: 19px;
            font-weight: 750;
            line-height: 1.4;
            letter-spacing: -0.3px;
            overflow-wrap: anywhere;
        }

        #interviewList .history-date {
            flex: 0 1 auto;
            color: #94a9cb;
            font-size: 12px;
            line-height: 1.6;
            overflow-wrap: anywhere;
        }

        #interviewList .history-details {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            width: 100%;
            min-width: 0;
        }

        #interviewList .history-field {
            display: grid;
            grid-template-columns: minmax(95px, 125px) minmax(0, 1fr);
            align-items: start;
            gap: 12px;
            width: 100%;
            min-width: 0;
            margin: 0;
            padding: 10px 0;
            border-bottom: 1px solid rgba(123, 147, 187, 0.13);
        }

        #interviewList .history-field:last-child {
            border-bottom: none;
        }

        #interviewList .history-field-label {
            display: block;
            min-width: 0;
            color: #8e9fbd;
            font-size: 12px;
            font-weight: 600;
            line-height: 1.7;
            overflow-wrap: anywhere;
        }

        #interviewList .history-field-value {
            display: block;
            min-width: 0;
            color: #e6edfb;
            font-size: 13px;
            font-weight: 650;
            line-height: 1.7;
            overflow-wrap: anywhere;
            word-break: normal;
        }

        #interviewList .history-field.score-field
        .history-field-value {
            color: #8bb2ff;
            font-weight: 800;
        }

        #interviewList .history-actions {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 17px;
            padding-top: 15px;
            border-top: 1px solid #25334b;
        }

        #interviewList .history-view-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 7px;
            max-width: 100%;
            min-height: 39px;
            padding: 9px 13px;
            color: #ffffff;
            background: linear-gradient(110deg, #704dff, #256df5);
            border: 1px solid transparent;
            border-radius: 9px;
            font-size: 12px;
            font-weight: 750;
            line-height: 1.5;
            text-align: center;
            text-decoration: none;
            overflow-wrap: anywhere;
            transition: filter 0.2s, transform 0.2s;
        }

        #interviewList .history-view-button:hover {
            filter: brightness(1.13);
            transform: translateY(-1px);
        }

        #interviewList .history-delete-button {
            min-height: 37px;
            padding: 8px 13px;
            color: #ffaaaa;
            background: rgba(133, 39, 52, 0.13);
            border: 1px solid #663441;
            border-radius: 9px;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
            transition: background 0.2s, border-color 0.2s;
        }

        #interviewList .history-delete-button:hover {
            background: rgba(170, 45, 62, 0.24);
            border-color: #a34b5c;
        }

        #interviewList .history-status {
            margin: 12px 0;
            padding: 18px;
            color: #a6b6d1;
            background: #101a2b;
            border: 1px solid #273650;
            border-radius: 12px;
            font-size: 13px;
            overflow-wrap: anywhere;
        }

        #interviewList .history-empty {
            padding: 28px 18px;
            color: #a6b6d1;
            background: #0d1524;
            border: 1px dashed #344765;
            border-radius: 13px;
            text-align: center;
        }

        #interviewList .history-empty p {
            margin: 0 0 17px;
            font-size: 13px;
            line-height: 1.8;
        }

        @media (max-width: 560px) {
            #interviewList .history-card {
                padding: 17px 14px;
                border-radius: 13px;
            }

            #interviewList .history-role {
                font-size: 17px;
            }

            #interviewList .history-date {
                flex-basis: 100%;
            }

            #interviewList .history-field {
                grid-template-columns: minmax(84px, 100px) minmax(0, 1fr);
                gap: 9px;
                padding: 9px 0;
            }

            #interviewList .history-field-label {
                font-size: 11px;
            }

            #interviewList .history-field-value {
                font-size: 12px;
            }

            #interviewList .history-actions {
                align-items: stretch;
                flex-direction: column;
            }

            #interviewList .history-view-button,
            #interviewList .history-delete-button {
                width: 100%;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            #interviewList .history-card,
            #interviewList .history-view-button {
                transition: none;
            }
        }
    `;

    document.head.appendChild(style);
}

/* =====================================================
   HISTORY FIELD
===================================================== */

function createField(label, value, className = "") {
    const field = makeElement(
        "div",
        `history-field ${className}`.trim()
    );

    const labelElement = makeElement(
        "span",
        "history-field-label",
        label
    );

    const valueElement = makeElement(
        "span",
        "history-field-value",
        value ?? "—"
    );

    field.append(labelElement, valueElement);

    return field;
}

/* =====================================================
   CREATE INTERVIEW CARD

   Data fields and result URL preserved
===================================================== */

function createInterviewCard(interview) {
    const card = makeElement("article", "history-card");

    const heading = makeElement("div", "history-heading");

    const role = makeElement(
        "h3",
        "history-role",
        interview.role || "Untitled Interview"
    );

    const date = makeElement(
        "span",
        "history-date",
        formatDate(interview.created_at)
    );

    heading.append(role, date);

    const details = makeElement("div", "history-details");

    details.append(
        createField(
            "Topic",
            interview.topic || "General"
        ),
        createField(
            "Difficulty",
            interview.difficulty || "—"
        ),
        createField(
            "Questions",
            Number(interview.total_questions || 0)
        ),
        createField(
            "Average Score",
            formatScore(interview.average_score),
            "score-field"
        )
    );

    const actions = makeElement("div", "history-actions");

    const viewLink = makeElement(
        "a",
        "history-view-button",
        "View Result & Model Answers →"
    );

    viewLink.href =
        `/interview/${encodeURIComponent(interview.id)}/result`;

    const deleteButton = makeElement(
        "button",
        "history-delete-button",
        "Delete"
    );

    deleteButton.type = "button";

    deleteButton.addEventListener("click", () => {
        deleteInterview(interview.id);
    });

    actions.append(viewLink, deleteButton);
    card.append(heading, details, actions);

    return card;
}

/* =====================================================
   LOAD STATISTICS
   Existing endpoint and IDs preserved
===================================================== */

async function loadStatistics() {
    try {
        const data = await fetchJSON("/api/statistics");

        const interviews = $("#totalInterviews");
        const questions = $("#totalQuestions");
        const average = $("#averageScore");

        if (interviews) {
            interviews.textContent = Number(
                data.total_interviews || 0
            ).toString();
        }

        if (questions) {
            questions.textContent = Number(
                data.total_questions || 0
            ).toString();
        }

        if (average) {
            average.textContent = formatScore(data.average_score);
        }
    } catch (error) {
        console.error("Statistics loading failed:", error);
    }
}

/* =====================================================
   LOAD INTERVIEW HISTORY
   Existing API response handling preserved
===================================================== */

async function loadInterviews() {
    const container = $("#interviewList");

    if (!container) return;

    container.replaceChildren(
        makeElement(
            "p",
            "history-status",
            "Loading interviews..."
        )
    );

    try {
        const data = await fetchJSON("/api/interviews");

        const interviews = Array.isArray(data.interviews)
            ? data.interviews
            : [];

        if (!interviews.length) {
            const empty = makeElement("div", "history-empty");

            empty.append(
                makeElement(
                    "p",
                    "",
                    "No interviews yet. Start a mock interview to see your history here."
                )
            );

            const link = makeElement(
                "a",
                "history-view-button",
                "Start New Interview →"
            );

            link.href = "/interview/setup";

            empty.append(link);
            container.replaceChildren(empty);

            return;
        }

        const fragment = document.createDocumentFragment();

        interviews.forEach((interview) => {
            fragment.append(createInterviewCard(interview));
        });

        container.replaceChildren(fragment);
    } catch (error) {
        console.error("Interview history loading failed:", error);

        container.replaceChildren(
            makeElement(
                "p",
                "history-status",
                error.message || "Could not load interview history."
            )
        );
    }
}

/* =====================================================
   DELETE INTERVIEW
   Existing delete API preserved
===================================================== */

async function deleteInterview(id) {
    if (!window.confirm("Delete this interview and its saved results?")) {
        return;
    }

    try {
        await fetchJSON(
            `/api/interviews/${encodeURIComponent(id)}`,
            { method: "DELETE" }
        );

        await Promise.all([
            loadStatistics(),
            loadInterviews()
        ]);
    } catch (error) {
        window.alert(
            error.message || "Could not delete the interview."
        );
    }
}

/* =====================================================
   INITIALIZE DASHBOARD
===================================================== */

document.addEventListener("DOMContentLoaded", () => {
    addDashboardStyles();
    loadStatistics();
    loadInterviews();
});
