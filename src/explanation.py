from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learner_shap_explanations_BBB_2014J_day60.csv"
)


# =========================================================
# Load inputs
# =========================================================

def load_inputs():
    """
    Load:
        - Day-60 learner profiles
        - integrated neuro-symbolic decisions
        - fitted Logistic Regression pipeline
    """

    required_files = [
        PROFILE_FILE,
        INTEGRATED_FILE,
        MODEL_FILE,
    ]

    for file in required_files:

        if not file.exists():
            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{file}"
            )

    profiles = pd.read_csv(
        PROFILE_FILE
    )

    integrated = pd.read_csv(
        INTEGRATED_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    return (
        profiles,
        integrated,
        model,
    )


# =========================================================
# Prepare transformed model input
# =========================================================

def prepare_shap_data(
    profiles,
    model,
):
    """
    Reuse the exact feature engineering expected by the
    trained model, then transform X using the saved
    preprocessing pipeline.

    This guarantees SHAP explains the same representation
    used by Logistic Regression.
    """

    (
        X,
        y,
        learner_ids,
        model_features,
    ) = prepare_features(
        profiles
    )

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    X_transformed = (
        preprocessor.transform(
            X
        )
    )

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
# Clean feature names
# =========================================================

def clean_feature_name(name):
    """
    Convert transformed names such as:

        numeric__days_since_last_activity

    into:

        days_since_last_activity
    """

    if "__" in name:
        return name.split(
            "__",
            1,
        )[1]

    return name


# =========================================================
# Build SHAP explainer
# =========================================================

def build_explainer(
    classifier,
    X_transformed,
):
    """
    Build a SHAP LinearExplainer for Logistic Regression.
    """

    explainer = shap.LinearExplainer(
        classifier,
        X_transformed,
    )

    return explainer


# =========================================================
# Calculate SHAP values
# =========================================================

def calculate_shap_values(
    explainer,
    X_transformed,
):
    """
    Calculate learner-level SHAP values.

    Positive SHAP value:
        pushes prediction toward Fail/Withdraw.

    Negative SHAP value:
        pushes prediction toward Pass/Distinction.
    """

    shap_values = explainer(
        X_transformed
    )

    return shap_values


# =========================================================
# Create learner-level explanations
# =========================================================

def build_learner_explanations(
    learner_ids,
    feature_names,
    shap_values,
    integrated,
    top_n=5,
):
    """
    Produce compact learner-level feature contribution
    explanations.

    For each learner:
        - strongest risk-increasing features
        - strongest risk-reducing features
    """

    cleaned_names = [
        clean_feature_name(
            name
        )
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
                    values[
                        index
                    ],
            }
        )

        # -------------------------------------------------
        # Risk-increasing contributions
        # -------------------------------------------------

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
            .head(
                top_n
            )
        )

        # -------------------------------------------------
        # Risk-reducing contributions
        # -------------------------------------------------

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
            .head(
                top_n
            )
        )

        positive_features = (
            "; ".join(
                [
                    (
                        f"{row.feature}="
                        f"{row.shap_value:.4f}"
                    )
                    for row
                    in positive.itertuples()
                ]
            )
        )

        negative_features = (
            "; ".join(
                [
                    (
                        f"{row.feature}="
                        f"{row.shap_value:.4f}"
                    )
                    for row
                    in negative.itertuples()
                ]
            )
        )

        rows.append(
            {
                "id_student":
                    int(
                        learner_id
                    ),

                "top_risk_increasing_features":
                    positive_features,

                "top_risk_reducing_features":
                    negative_features,
            }
        )

    explanations = pd.DataFrame(
        rows
    )

    # -----------------------------------------------------
    # Attach integrated decision context
    # -----------------------------------------------------

    context_columns = [
        "id_student",
        "ml_risk_probability",
        "ml_risk_band",
        "risk_state",
        "final_priority",
        "final_intervention",
        "integration_rule",
        "agreement_state",
    ]

    explanations = explanations.merge(
        integrated[
            context_columns
        ],
        on="id_student",
        how="left",
        validate="one_to_one",
    )

    return explanations


# =========================================================
# Validation
# =========================================================

def validate_explanations(
    explanations,
):
    """
    Validate SHAP explanation coverage.
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

    print(
        f"Learners explained: "
        f"{len(explanations):,}"
    )

    print(
        f"Duplicate learners: "
        f"{explanations['id_student'].duplicated().sum()}"
    )

    print(
        f"Missing ML probabilities: "
        f"{explanations['ml_risk_probability'].isna().sum()}"
    )

    print(
        f"Missing final priorities: "
        f"{explanations['final_priority'].isna().sum()}"
    )

    if (
        explanations[
            "id_student"
        ]
        .duplicated()
        .sum()
        == 0
        and explanations[
            "ml_risk_probability"
        ]
        .isna()
        .sum()
        == 0
    ):

        print(
            "\nSHAP validation: PASS"
        )

    else:

        print(
            "\nSHAP validation: FAIL"
        )


# =========================================================
# Print example explanations
# =========================================================

def print_examples(
    explanations,
    n=10,
):
    """
    Print representative SHAP explanations.
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
        .head(
            n
        )
        .to_string(
            index=False
        )
    )


# =========================================================
# Global SHAP importance
# =========================================================

def calculate_global_importance(
    feature_names,
    shap_values,
):
    """
    Calculate global feature importance using mean absolute
    SHAP values.
    """

    cleaned_names = [
        clean_feature_name(
            name
        )
        for name in feature_names
    ]

    mean_absolute = (
        np.abs(
            shap_values.values
        )
        .mean(
            axis=0
        )
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
        .reset_index(
            drop=True
        )
    )

    return importance


# =========================================================
# Save outputs
# =========================================================

def save_outputs(
    explanations,
    global_importance,
):
    """
    Save local and global SHAP explanation results.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    explanations.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    global_file = (
        OUTPUT_FILE.parent
        / "global_shap_importance_BBB_2014J_day60.csv"
    )

    global_importance.to_csv(
        global_file,
        index=False,
    )

    print(
        "\nSaved learner SHAP explanations:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nSaved global SHAP importance:"
    )

    print(
        global_file
    )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print(
        "\nGenerating SHAP explanations..."
    )

    # -----------------------------------------------------
    # 1. Load inputs
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
    # 2. Prepare model representation
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
    # 3. Create SHAP explainer
    # -----------------------------------------------------

    explainer = build_explainer(
        classifier,
        X_transformed,
    )

    # -----------------------------------------------------
    # 4. Calculate SHAP
    # -----------------------------------------------------

    shap_values = calculate_shap_values(
        explainer,
        X_transformed,
    )

    print(
        f"SHAP explanations generated: "
        f"{len(shap_values.values):,}"
    )

    # -----------------------------------------------------
    # 5. Learner-level explanations
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
    # 6. Global feature importance
    # -----------------------------------------------------

    global_importance = (
        calculate_global_importance(
            feature_names,
            shap_values,
        )
    )

    # -----------------------------------------------------
    # 7. Validate
    # -----------------------------------------------------

    validate_explanations(
        explanations
    )

    # -----------------------------------------------------
    # 8. Display examples
    # -----------------------------------------------------

    print_examples(
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
    # 9. Save
    # -----------------------------------------------------

    save_outputs(
        explanations,
        global_importance,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SHAP EXPLANATION GENERATION COMPLETE"
    )

    print(
        "=" * 70
    )