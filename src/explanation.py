from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import shap

from openai import OpenAI

from src.model import prepare_features


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROFILE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learner_profiles_BBB_2014J_day60.csv"
)

INTEGRATED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learner_integrated_decisions_BBB_2014J_day60.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "logistic_regression_day60.joblib"
)

SHAP_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learner_shap_explanations_BBB_2014J_day60.csv"
)

GLOBAL_SHAP_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "global_shap_importance_BBB_2014J_day60.csv"
)

LLM_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learner_llm_explanations_BBB_2014J_day60.csv"
)


# =========================================================
# OpenAI configuration
# =========================================================

OPENAI_MODEL = "gpt-5.5"


# =========================================================
# Load inputs
# =========================================================

def load_inputs():
    """
    Load:
        - Day-60 learner profiles
        - integrated neuro-symbolic decisions
        - trained Logistic Regression model
    """

    required_files = [
        PROFILE_FILE,
        INTEGRATED_FILE,
        MODEL_FILE,
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{file}"
            )

    profiles = pd.read_csv(PROFILE_FILE)
    integrated = pd.read_csv(INTEGRATED_FILE)
    model = joblib.load(MODEL_FILE)

    return profiles, integrated, model


# =========================================================
# Prepare SHAP data
# =========================================================

def prepare_shap_data(
    profiles,
    model,
):
    """
    Prepare exactly the same transformed features used by
    the trained Logistic Regression model.
    """

    (
        X,
        y,
        learner_ids,
        model_features,
    ) = prepare_features(profiles)

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    X_transformed = preprocessor.transform(X)

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    return (
        X,
        X_transformed,
        learner_ids,
        classifier,
        feature_names,
    )


# =========================================================
# Feature-name helper
# =========================================================

def clean_feature_name(name):
    """
    Convert:

        numeric__days_since_last_activity

    into:

        days_since_last_activity
    """

    if "__" in name:
        return name.split("__", 1)[1]

    return name


# =========================================================
# Build SHAP explainer
# =========================================================

def build_explainer(
    classifier,
    X_transformed,
):
    """
    Build a SHAP LinearExplainer suitable for Logistic
    Regression.
    """

    explainer = shap.LinearExplainer(
        classifier,
        X_transformed,
    )

    return explainer


# =========================================================
# Calculate SHAP
# =========================================================

def calculate_shap_values(
    explainer,
    X_transformed,
):
    """
    Positive SHAP values push toward Fail/Withdraw.
    Negative SHAP values push toward Pass/Distinction.
    """

    return explainer(X_transformed)


# =========================================================
# Learner-level SHAP explanations
# =========================================================

def build_learner_explanations(
    learner_ids,
    feature_names,
    shap_values,
    integrated,
    top_n=5,
):
    """
    Build one compact SHAP explanation per learner.
    """

    cleaned_names = [
        clean_feature_name(name)
        for name in feature_names
    ]

    rows = []

    values = shap_values.values

    for index, learner_id in enumerate(
        learner_ids.values
    ):

        contributions = pd.DataFrame(
            {
                "feature":
                    cleaned_names,

                "shap_value":
                    values[index],
            }
        )

        positive = (
            contributions[
                contributions[
                    "shap_value"
                ] > 0
            ]
            .sort_values(
                "shap_value",
                ascending=False,
            )
            .head(top_n)
        )

        negative = (
            contributions[
                contributions[
                    "shap_value"
                ] < 0
            ]
            .sort_values(
                "shap_value",
                ascending=True,
            )
            .head(top_n)
        )

        positive_features = "; ".join(
            [
                (
                    f"{row.feature}="
                    f"{row.shap_value:.4f}"
                )
                for row
                in positive.itertuples()
            ]
        )

        negative_features = "; ".join(
            [
                (
                    f"{row.feature}="
                    f"{row.shap_value:.4f}"
                )
                for row
                in negative.itertuples()
            ]
        )

        rows.append(
            {
                "id_student":
                    int(learner_id),

                "top_risk_increasing_features":
                    positive_features,

                "top_risk_reducing_features":
                    negative_features,
            }
        )

    explanations = pd.DataFrame(rows)

    context_columns = [
        "id_student",
        "ml_risk_probability",
        "ml_risk_band",
        "weighted_assessment_average",
        "assessment_trend",
        "recent_14_day_clicks",
        "previous_14_day_clicks",
        "engagement_change",
        "days_since_last_activity",
        "assessment_completion_rate",
        "performance_state",
        "assessment_trend_state",
        "engagement_state",
        "inactivity_state",
        "completion_state",
        "evidence_sufficiency",
        "risk_state",
        "intervention",
        "rule_id",
        "rule_explanation",
        "final_priority",
        "final_intervention",
        "integration_rule",
        "agreement_state",
        "integration_explanation",
    ]

    explanations = explanations.merge(
        integrated[context_columns],
        on="id_student",
        how="left",
        validate="one_to_one",
    )

    return explanations


# =========================================================
# Global SHAP importance
# =========================================================

def calculate_global_importance(
    feature_names,
    shap_values,
):
    """
    Calculate global mean absolute SHAP importance.
    """

    cleaned_names = [
        clean_feature_name(name)
        for name in feature_names
    ]

    mean_absolute = (
        np.abs(shap_values.values)
        .mean(axis=0)
    )

    importance = pd.DataFrame(
        {
            "feature":
                cleaned_names,

            "mean_absolute_shap":
                mean_absolute,
        }
    )

    importance = (
        importance
        .sort_values(
            "mean_absolute_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return importance


# =========================================================
# SHAP validation
# =========================================================

def validate_explanations(
    explanations,
):
    """
    Validate learner-level SHAP coverage.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SHAP EXPLANATION VALIDATION"
    )

    print(
        "=" * 70
    )

    duplicate_count = (
        explanations[
            "id_student"
        ]
        .duplicated()
        .sum()
    )

    missing_ml = (
        explanations[
            "ml_risk_probability"
        ]
        .isna()
        .sum()
    )

    missing_priority = (
        explanations[
            "final_priority"
        ]
        .isna()
        .sum()
    )

    print(
        f"Learners explained: "
        f"{len(explanations):,}"
    )

    print(
        f"Duplicate learners: "
        f"{duplicate_count}"
    )

    print(
        f"Missing ML probabilities: "
        f"{missing_ml}"
    )

    print(
        f"Missing final priorities: "
        f"{missing_priority}"
    )

    if (
        duplicate_count == 0
        and missing_ml == 0
        and missing_priority == 0
    ):

        print(
            "\nSHAP validation: PASS"
        )

    else:

        print(
            "\nSHAP validation: FAIL"
        )


# =========================================================
# Print SHAP examples
# =========================================================

def print_shap_examples(
    explanations,
    n=10,
):
    """
    Display representative learner-level SHAP results.
    """

    columns = [
        "id_student",
        "ml_risk_probability",
        "ml_risk_band",
        "top_risk_increasing_features",
        "top_risk_reducing_features",
        "risk_state",
        "final_priority",
        "final_intervention",
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXAMPLE SHAP EXPLANATIONS"
    )

    print(
        "=" * 70
    )

    print(
        explanations[
            columns
        ]
        .head(n)
        .to_string(index=False)
    )


# =========================================================
# OpenAI client
# =========================================================

def get_openai_client():
    """
    Return an OpenAI client using the temporary environment
    variable OPENAI_API_KEY.

    The API key must not be hard-coded into this project.
    """

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        return None

    return OpenAI(
        api_key=api_key
    )


# =========================================================
# Build grounded learner evidence
# =========================================================

def build_llm_evidence(
    learner_row,
):
    """
    Construct a structured textual representation of the
    already-established learner evidence.

    The LLM receives conclusions generated by the system.
    It does not receive raw OULAD records.
    """

    evidence = f"""
Learner observation point:
Day 60

Predictive model:
Risk probability: {learner_row['ml_risk_probability']:.3f}
ML risk band: {learner_row['ml_risk_band']}

Numerical evidence:
Weighted assessment average: {learner_row['weighted_assessment_average']}
Assessment trend: {learner_row['assessment_trend']}
Recent 14-day clicks: {learner_row['recent_14_day_clicks']}
Previous 14-day clicks: {learner_row['previous_14_day_clicks']}
Engagement change: {learner_row['engagement_change']}
Days since last activity: {learner_row['days_since_last_activity']}
Assessment completion rate: {learner_row['assessment_completion_rate']}

Semantic evidence:
Performance state: {learner_row['performance_state']}
Assessment trend state: {learner_row['assessment_trend_state']}
Engagement state: {learner_row['engagement_state']}
Inactivity state: {learner_row['inactivity_state']}
Completion state: {learner_row['completion_state']}
Evidence sufficiency: {learner_row['evidence_sufficiency']}

Symbolic reasoning:
Symbolic risk state: {learner_row['risk_state']}
Symbolic intervention: {learner_row['intervention']}
Symbolic rule: {learner_row['rule_id']}
Symbolic rule explanation:
{learner_row['rule_explanation']}

Neuro-symbolic integration:
Final priority: {learner_row['final_priority']}
Final intervention: {learner_row['final_intervention']}
Integration rule: {learner_row['integration_rule']}
Agreement state: {learner_row['agreement_state']}
Integration explanation:
{learner_row['integration_explanation']}

SHAP risk-increasing factors:
{learner_row['top_risk_increasing_features']}

SHAP risk-reducing factors:
{learner_row['top_risk_reducing_features']}
"""

    return evidence.strip()


# =========================================================
# Generate one grounded LLM explanation
# =========================================================

def generate_llm_explanation(
    learner_row,
    client,
):
    """
    Generate an educator-facing explanation.

    The LLM is strictly a communication layer.

    It must not:
        - change the decision
        - invent evidence
        - infer motivation
        - infer personal circumstances
        - diagnose the learner
        - create a new intervention
    """

    evidence = build_llm_evidence(
        learner_row
    )

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "You are an explanation component inside an "
            "educational decision-support prototype. "

            "Write a concise educator-facing explanation "
            "using only the supplied structured evidence. "

            "Do not invent learner circumstances, motives, "
            "behavioural causes, diagnoses, personal details, "
            "or academic facts that are not explicitly "
            "provided. "

            "Do not change the supplied ML probability, "
            "symbolic risk state, final priority, or final "
            "intervention. "

            "Clearly distinguish observed evidence from model "
            "prediction. "

            "If evidence is partial or insufficient, say so. "

            "If probabilistic and symbolic components disagree "
            "or show mixed evidence, explicitly mention that. "

            "Present the recommendation as decision support "
            "for an educator and not as an automatic or final "
            "determination. "

            "Use approximately 80 to 130 words."
        ),
        input=evidence,
    )

    return response.output_text


# =========================================================
# Select representative learners
# =========================================================

def select_llm_examples(
    explanations,
):
    """
    Select one learner from each important final-priority
    category.

    This avoids making 2,292 API calls during prototype
    development.
    """

    priorities = [
        "HighPriority",
        "HumanReviewPriority",
        "ModeratePriority",
        "LowPriority",
    ]

    selected_rows = []

    for priority in priorities:

        subset = explanations[
            explanations[
                "final_priority"
            ] == priority
        ]

        if subset.empty:
            continue

        # Prefer the highest ML probability within the
        # category for a clear demonstration example.

        example = (
            subset
            .sort_values(
                "ml_risk_probability",
                ascending=False,
            )
            .iloc[0]
        )

        selected_rows.append(
            example
        )

    if not selected_rows:
        return pd.DataFrame()

    return pd.DataFrame(
        selected_rows
    ).reset_index(
        drop=True
    )


# =========================================================
# Generate representative LLM explanations
# =========================================================

def generate_representative_llm_explanations(
    explanations,
):
    """
    Generate LLM explanations for a small representative
    subset of learners.

    The function returns an empty DataFrame if no API key is
    configured.
    """

    client = get_openai_client()

    if client is None:

        print(
            "\nOPENAI_API_KEY is not configured."
        )

        print(
            "Skipping LLM explanation generation."
        )

        return pd.DataFrame()

    selected = select_llm_examples(
        explanations
    )

    rows = []

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GROUNDED LLM EDUCATOR EXPLANATIONS"
    )

    print(
        "=" * 70
    )

    for _, learner in selected.iterrows():

        try:

            explanation = (
                generate_llm_explanation(
                    learner,
                    client,
                )
            )

        except Exception as exc:

            explanation = (
                "LLM explanation generation failed: "
                f"{exc}"
            )

        print(
            f"\nLearner: "
            f"{int(learner['id_student'])}"
        )

        print(
            f"Priority: "
            f"{learner['final_priority']}"
        )

        print(
            f"Risk probability: "
            f"{learner['ml_risk_probability']:.3f}"
        )

        print()

        print(
            explanation
        )

        rows.append(
            {
                "id_student":
                    int(
                        learner[
                            "id_student"
                        ]
                    ),

                "ml_risk_probability":
                    learner[
                        "ml_risk_probability"
                    ],

                "final_priority":
                    learner[
                        "final_priority"
                    ],

                "final_intervention":
                    learner[
                        "final_intervention"
                    ],

                "agreement_state":
                    learner[
                        "agreement_state"
                    ],

                "llm_explanation":
                    explanation,
            }
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# Save outputs
# =========================================================

def save_outputs(
    explanations,
    global_importance,
    llm_explanations,
):
    """
    Save SHAP and LLM explanation outputs.
    """

    SHAP_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    explanations.to_csv(
        SHAP_OUTPUT_FILE,
        index=False,
    )

    global_importance.to_csv(
        GLOBAL_SHAP_OUTPUT_FILE,
        index=False,
    )

    print(
        "\nSaved learner SHAP explanations:"
    )

    print(
        SHAP_OUTPUT_FILE
    )

    print(
        "\nSaved global SHAP importance:"
    )

    print(
        GLOBAL_SHAP_OUTPUT_FILE
    )

    if not llm_explanations.empty:

        llm_explanations.to_csv(
            LLM_OUTPUT_FILE,
            index=False,
        )

        print(
            "\nSaved representative LLM explanations:"
        )

        print(
            LLM_OUTPUT_FILE
        )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print(
        "\nGenerating explainable learner decisions..."
    )

    # -----------------------------------------------------
    # 1. Load existing model and decision outputs
    # -----------------------------------------------------

    (
        profiles,
        integrated,
        model,
    ) = load_inputs()

    print(
        f"\nLearner profiles loaded: "
        f"{len(profiles):,}"
    )

    # -----------------------------------------------------
    # 2. Prepare transformed ML features
    # -----------------------------------------------------

    (
        X,
        X_transformed,
        learner_ids,
        classifier,
        feature_names,
    ) = prepare_shap_data(
        profiles,
        model,
    )

    print(
        f"Transformed model features: "
        f"{len(feature_names)}"
    )

    # -----------------------------------------------------
    # 3. Build SHAP explainer
    # -----------------------------------------------------

    explainer = build_explainer(
        classifier,
        X_transformed,
    )

    # -----------------------------------------------------
    # 4. Calculate SHAP values
    # -----------------------------------------------------

    shap_values = (
        calculate_shap_values(
            explainer,
            X_transformed,
        )
    )

    print(
        f"SHAP explanations generated: "
        f"{len(shap_values.values):,}"
    )

    # -----------------------------------------------------
    # 5. Build learner-level SHAP explanations
    # -----------------------------------------------------

    explanations = (
        build_learner_explanations(
            learner_ids=learner_ids,
            feature_names=feature_names,
            shap_values=shap_values,
            integrated=integrated,
            top_n=5,
        )
    )

    # -----------------------------------------------------
    # 6. Calculate global SHAP importance
    # -----------------------------------------------------

    global_importance = (
        calculate_global_importance(
            feature_names,
            shap_values,
        )
    )

    # -----------------------------------------------------
    # 7. Validate SHAP coverage
    # -----------------------------------------------------

    validate_explanations(
        explanations
    )

    # -----------------------------------------------------
    # 8. Print SHAP examples
    # -----------------------------------------------------

    print_shap_examples(
        explanations,
        n=10,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GLOBAL SHAP FEATURE IMPORTANCE"
    )

    print(
        "=" * 70
    )

    print(
        global_importance
        .head(15)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 9. Generate representative LLM explanations
    #
    # Only four representative learners are sent to the
    # API during prototype development.
    # -----------------------------------------------------

    llm_explanations = (
        generate_representative_llm_explanations(
            explanations
        )
    )

    # -----------------------------------------------------
    # 10. Save outputs
    # -----------------------------------------------------

    save_outputs(
        explanations=explanations,
        global_importance=global_importance,
        llm_explanations=llm_explanations,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXPLANATION PIPELINE COMPLETE"
    )

    print(
        "=" * 70
    )