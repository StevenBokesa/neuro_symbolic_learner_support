from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# Configuration
# =========================================================

st.set_page_config(
    page_title="Neuro-Symbolic Learner Support",
    page_icon="🎓",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parent

INTEGRATED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learner_integrated_decisions_BBB_2014J_day60.csv"
)

SHAP_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learner_shap_explanations_BBB_2014J_day60.csv"
)

LLM_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learner_llm_explanations_BBB_2014J_day60.csv"
)


# =========================================================
# Data loading
# =========================================================

@st.cache_data
def load_data():

    if not INTEGRATED_FILE.exists():
        st.error(
            f"Integrated decision file not found:\n"
            f"{INTEGRATED_FILE}"
        )
        st.stop()

    if not SHAP_FILE.exists():
        st.error(
            f"SHAP explanation file not found:\n"
            f"{SHAP_FILE}"
        )
        st.stop()

    integrated = pd.read_csv(
        INTEGRATED_FILE
    )

    shap = pd.read_csv(
        SHAP_FILE
    )

    # SHAP output already contains substantial context.
    # We only keep explanation-specific columns here to
    # avoid duplicate columns during the merge.

    shap_columns = [
        "id_student",
        "top_risk_increasing_features",
        "top_risk_reducing_features",
    ]

    data = integrated.merge(
        shap[shap_columns],
        on="id_student",
        how="left",
        validate="one_to_one",
    )

    # LLM explanations exist only for representative
    # learners at this stage.

    if LLM_FILE.exists():

        llm = pd.read_csv(
            LLM_FILE
        )

        llm_columns = [
            "id_student",
            "llm_explanation",
        ]

        data = data.merge(
            llm[llm_columns],
            on="id_student",
            how="left",
            validate="one_to_one",
        )

    else:

        data[
            "llm_explanation"
        ] = pd.NA

    return data


# =========================================================
# Display helpers
# =========================================================

def readable_label(value):

    if pd.isna(value):
        return "Not available"

    replacements = {
        "HighPriority": "High Priority",
        "ModeratePriority": "Moderate Priority",
        "LowPriority": "Low Priority",
        "HumanReviewPriority": "Human Review Priority",
        "MonitoringPriority": "Monitoring Priority",

        "HighRisk": "High Risk",
        "ModerateRisk": "Moderate Risk",
        "LowRisk": "Low Risk",
        "UncertainRisk": "Uncertain Risk",

        "TutorReview": "Tutor Review",
        "ContactLearner": "Contact Learner",
        "AutomatedReminder": "Automated Reminder",
        "ContinueMonitoring": "Continue Monitoring",

        "SufficientEvidence": "Sufficient Evidence",
        "PartialEvidence": "Partial Evidence",
        "InsufficientEvidence": "Insufficient Evidence",
    }

    return replacements.get(
        str(value),
        str(value),
    )


def format_number(
    value,
    digits=1,
):

    if pd.isna(value):
        return "Not available"

    return f"{value:.{digits}f}"


def format_percentage(value):

    if pd.isna(value):
        return "Not available"

    return f"{value * 100:.1f}%"


def split_shap_features(value):
    """
    Convert the stored semicolon-separated SHAP feature
    string into individual display items.
    """

    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]


# =========================================================
# Load data
# =========================================================

data = load_data()


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title(
    "Learner Selection"
)

st.sidebar.caption(
    "Experiment: BBB 2014J"
)

st.sidebar.caption(
    "Observation point: Day 60"
)


priority_options = [
    "All",
] + sorted(
    data[
        "final_priority"
    ]
    .dropna()
    .unique()
    .tolist()
)


selected_priority = st.sidebar.selectbox(
    "Filter by final priority",
    priority_options,
)


filtered_data = data.copy()


if selected_priority != "All":

    filtered_data = filtered_data[
        filtered_data[
            "final_priority"
        ] == selected_priority
    ]


learner_ids = (
    filtered_data[
        "id_student"
    ]
    .astype(int)
    .sort_values()
    .tolist()
)


if not learner_ids:

    st.warning(
        "No learners match the selected filter."
    )

    st.stop()


selected_learner_id = (
    st.sidebar.selectbox(
        "Select learner",
        learner_ids,
    )
)


learner = (
    filtered_data[
        filtered_data[
            "id_student"
        ] == selected_learner_id
    ]
    .iloc[0]
)


# =========================================================
# Header
# =========================================================

st.title(
    "Explainable Neuro-Symbolic Learner Support"
)

st.caption(
    "Educational decision-support prototype — "
    "OULAD BBB 2014J, Day 60"
)

st.info(
    "This prototype provides decision support for educators. "
    "Risk estimates and interventions should not be treated "
    "as automatic or final decisions."
)


st.subheader(
    f"Learner {selected_learner_id}"
)


# =========================================================
# Main decision metrics
# =========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "ML Risk Probability",
        format_percentage(
            learner[
                "ml_risk_probability"
            ]
        ),
    )


with col2:

    st.metric(
        "Symbolic Risk",
        readable_label(
            learner[
                "risk_state"
            ]
        ),
    )


with col3:

    st.metric(
        "Final Priority",
        readable_label(
            learner[
                "final_priority"
            ]
        ),
    )


with col4:

    st.metric(
        "Intervention",
        readable_label(
            learner[
                "final_intervention"
            ]
        ),
    )


st.divider()


# =========================================================
# Evidence
# =========================================================

st.header(
    "1. Learner Evidence"
)


left, right = st.columns(2)


with left:

    st.subheader(
        "Assessment Evidence"
    )

    st.write(
        "**Weighted assessment average:**",
        format_number(
            learner[
                "weighted_assessment_average"
            ]
        ),
    )

    st.write(
        "**Assessment trend:**",
        format_number(
            learner[
                "assessment_trend"
            ]
        ),
    )

    st.write(
        "**Assessment completion rate:**",
        format_percentage(
            learner[
                "assessment_completion_rate"
            ]
        ),
    )

    st.write(
        "**Performance state:**",
        readable_label(
            learner[
                "performance_state"
            ]
        ),
    )

    st.write(
        "**Assessment trend state:**",
        readable_label(
            learner[
                "assessment_trend_state"
            ]
        ),
    )

    st.write(
        "**Completion state:**",
        readable_label(
            learner[
                "completion_state"
            ]
        ),
    )


with right:

    st.subheader(
        "Engagement Evidence"
    )

    st.write(
        "**Recent 14-day clicks:**",
        format_number(
            learner[
                "recent_14_day_clicks"
            ],
            digits=0,
        ),
    )

    st.write(
        "**Previous 14-day clicks:**",
        format_number(
            learner[
                "previous_14_day_clicks"
            ],
            digits=0,
        ),
    )

    st.write(
        "**Engagement change:**",
        format_number(
            learner[
                "engagement_change"
            ],
            digits=2,
        ),
    )

    st.write(
        "**Days since last activity:**",
        format_number(
            learner[
                "days_since_last_activity"
            ],
            digits=0,
        ),
    )

    st.write(
        "**Engagement state:**",
        readable_label(
            learner[
                "engagement_state"
            ]
        ),
    )

    st.write(
        "**Inactivity state:**",
        readable_label(
            learner[
                "inactivity_state"
            ]
        ),
    )


st.write(
    "**Evidence sufficiency:**",
    readable_label(
        learner[
            "evidence_sufficiency"
        ]
    ),
)


st.divider()


# =========================================================
# ML explanation
# =========================================================

st.header(
    "2. Predictive Model Explanation"
)


st.write(
    "The Logistic Regression model estimates:"
)

st.metric(
    "Probability of Fail / Withdraw",
    format_percentage(
        learner[
            "ml_risk_probability"
        ]
    ),
)


risk_features = split_shap_features(
    learner[
        "top_risk_increasing_features"
    ]
)

protective_features = split_shap_features(
    learner[
        "top_risk_reducing_features"
    ]
)


shap_left, shap_right = st.columns(2)


with shap_left:

    st.subheader(
        "Factors Increasing Predicted Risk"
    )

    if risk_features:

        for feature in risk_features:
            st.write(
                f"↑ {feature}"
            )

    else:

        st.write(
            "No positive SHAP contributions available."
        )


with shap_right:

    st.subheader(
        "Factors Reducing Predicted Risk"
    )

    if protective_features:

        for feature in protective_features:
            st.write(
                f"↓ {feature}"
            )

    else:

        st.write(
            "No negative SHAP contributions available."
        )


st.caption(
    "SHAP values explain how each model feature moved this "
    "learner's prediction relative to the model's reference "
    "prediction. They do not establish causal effects."
)


st.divider()


# =========================================================
# Symbolic reasoning
# =========================================================

st.header(
    "3. Symbolic Reasoning"
)


symbolic1, symbolic2, symbolic3 = (
    st.columns(3)
)


with symbolic1:

    st.metric(
        "Symbolic Risk",
        readable_label(
            learner[
                "risk_state"
            ]
        ),
    )


with symbolic2:

    st.metric(
        "Rule",
        str(
            learner[
                "rule_id"
            ]
        ),
    )


with symbolic3:

    st.metric(
        "Symbolic Intervention",
        readable_label(
            learner[
                "intervention"
            ]
        ),
    )


st.subheader(
    "Rule Explanation"
)

st.write(
    learner[
        "rule_explanation"
    ]
)


st.divider()


# =========================================================
# Neuro-symbolic integration
# =========================================================

st.header(
    "4. Integrated Decision"
)


decision1, decision2, decision3 = (
    st.columns(3)
)


with decision1:

    st.metric(
        "Final Priority",
        readable_label(
            learner[
                "final_priority"
            ]
        ),
    )


with decision2:

    st.metric(
        "Final Intervention",
        readable_label(
            learner[
                "final_intervention"
            ]
        ),
    )


with decision3:

    st.metric(
        "Integration Rule",
        str(
            learner[
                "integration_rule"
            ]
        ),
    )


st.write(
    "**Agreement state:**",
    readable_label(
        learner[
            "agreement_state"
        ]
    ),
)


st.subheader(
    "Integration Explanation"
)

st.write(
    learner[
        "integration_explanation"
    ]
)


st.divider()


# =========================================================
# Grounded LLM explanation
# =========================================================

st.header(
    "5. Educator-Facing Explanation"
)


llm_explanation = learner.get(
    "llm_explanation"
)


if (
    pd.notna(llm_explanation)
    and str(llm_explanation).strip()
):

    st.success(
        llm_explanation
    )

else:

    st.info(
        "A grounded LLM explanation has not been generated "
        "for this learner. The underlying ML, SHAP, semantic "
        "and symbolic explanations remain available above."
    )


st.caption(
    "The language-model layer communicates evidence already "
    "produced by the framework. It does not determine the "
    "risk probability, symbolic state, priority or intervention."
)


# =========================================================
# Technical details
# =========================================================

with st.expander(
    "Technical decision details"
):

    technical_fields = {
        "Learner":
            int(
                learner[
                    "id_student"
                ]
            ),

        "ML risk probability":
            learner[
                "ml_risk_probability"
            ],

        "ML risk band":
            learner[
                "ml_risk_band"
            ],

        "Symbolic risk":
            learner[
                "risk_state"
            ],

        "Symbolic rule":
            learner[
                "rule_id"
            ],

        "Integration rule":
            learner[
                "integration_rule"
            ],

        "Agreement state":
            learner[
                "agreement_state"
            ],

        "Final priority":
            learner[
                "final_priority"
            ],

        "Final intervention":
            learner[
                "final_intervention"
            ],
    }

    st.json(
        technical_fields
    )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "Explainable Neuro-Symbolic Learner Support Framework | "
    "OULAD BBB 2014J | Prediction point: Day 60"
)