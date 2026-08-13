from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learner_profiles_BBB_2014J_day60.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

TEST_SIZE = 0.20
RANDOM_STATE = 42

TARGET_COLUMN = "risk_target"


# ---------------------------------------------------------
# Base model features
# ---------------------------------------------------------

BASE_FEATURES = [
    "weighted_assessment_average",
    "latest_assessment_score",
    "assessment_trend",
    "assessments_submitted",
    "assessment_completion_rate",
    "late_submissions",
    "total_clicks",
    "recent_14_day_clicks",
    "previous_14_day_clicks",
    "engagement_change",
    "active_days",
    "days_since_last_activity",
    "previous_attempts",
    "studied_credits",
]


# ---------------------------------------------------------
# Load learner profiles
# ---------------------------------------------------------

def load_profiles():
    """
    Load the Day-60 learner-profile dataset.
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed learner-profile file not found:\n"
            f"{DATA_FILE}\n\n"
            f"Run:\n"
            f"python -m src.feature_engineering"
        )

    profiles = pd.read_csv(
        DATA_FILE
    )

    return profiles


# ---------------------------------------------------------
# Add evidence-availability indicators
# ---------------------------------------------------------

def add_evidence_indicators(
    profiles,
):
    """
    Add explicit binary indicators that distinguish
    missing evidence from genuinely poor performance.

    This is important because:

        missing assessment score != score of zero

    Indicators:
        has_assessment_average
        has_latest_assessment_score
        has_assessment_trend
        has_vle_activity
        has_recent_vle_activity
    """

    df = profiles.copy()

    df[
        "has_assessment_average"
    ] = (
        df[
            "weighted_assessment_average"
        ]
        .notna()
        .astype(int)
    )

    df[
        "has_latest_assessment_score"
    ] = (
        df[
            "latest_assessment_score"
        ]
        .notna()
        .astype(int)
    )

    df[
        "has_assessment_trend"
    ] = (
        df[
            "assessment_trend"
        ]
        .notna()
        .astype(int)
    )

    df[
        "has_vle_activity"
    ] = (
        df[
            "total_clicks"
        ] > 0
    ).astype(int)

    df[
        "has_recent_vle_activity"
    ] = (
        df[
            "recent_14_day_clicks"
        ] > 0
    ).astype(int)

    return df


# ---------------------------------------------------------
# Stabilise engagement change
# ---------------------------------------------------------

def stabilise_engagement_change(
    profiles,
):
    """
    Clip extreme engagement-change ratios.

    The raw feature may become extremely large when the
    previous 14-day click count is close to zero.

    Clipping preserves direction while preventing rare
    extreme values from dominating Logistic Regression.

    The original column is retained separately for later
    analysis if needed.
    """

    df = profiles.copy()

    df[
        "engagement_change_raw"
    ] = df[
        "engagement_change"
    ]

    df[
        "engagement_change"
    ] = (
        df[
            "engagement_change"
        ]
        .clip(
            lower=-1.0,
            upper=5.0,
        )
    )

    return df


# ---------------------------------------------------------
# Prepare X and y
# ---------------------------------------------------------

def prepare_features(
    profiles,
):
    """
    Build model input matrix X and target vector y.

    Excluded:
        id_student
        final_result
        risk_target

    final_result is future outcome information and must never
    appear in X.
    """

    profiles = add_evidence_indicators(
        profiles
    )

    profiles = stabilise_engagement_change(
        profiles
    )

    indicator_features = [
        "has_assessment_average",
        "has_latest_assessment_score",
        "has_assessment_trend",
        "has_vle_activity",
        "has_recent_vle_activity",
    ]

    model_features = (
        BASE_FEATURES
        + indicator_features
    )

    missing_columns = [
        column
        for column in model_features
        if column not in profiles.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following expected model features "
            f"are missing:\n{missing_columns}"
        )

    X = profiles[
        model_features
    ].copy()

    y = profiles[
        TARGET_COLUMN
    ].copy()

    learner_ids = profiles[
        "id_student"
    ].copy()

    return (
        X,
        y,
        learner_ids,
        model_features,
    )


# ---------------------------------------------------------
# Build Logistic Regression pipeline
# ---------------------------------------------------------

def build_model_pipeline(
    model_features,
):
    """
    Build preprocessing + Logistic Regression pipeline.

    Continuous/numeric features:
        median imputation
        standard scaling

    Evidence-indicator features:
        most-frequent imputation
        no scaling required
    """

    indicator_features = [
        "has_assessment_average",
        "has_latest_assessment_score",
        "has_assessment_trend",
        "has_vle_activity",
        "has_recent_vle_activity",
    ]

    numeric_features = [
        feature
        for feature in model_features
        if feature
        not in indicator_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    indicator_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "indicators",
                indicator_pipeline,
                indicator_features,
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return pipeline


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate binary classifier using several metrics.

    Accuracy alone is not sufficient.
    """

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOGISTIC REGRESSION EVALUATION"
    )

    print(
        "=" * 70
    )

    for name, value in metrics.items():
        print(
            f"{name:<12}: "
            f"{value:.4f}"
        )

    print(
        "\nConfusion matrix"
    )

    print("-" * 50)

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print(
        matrix
    )

    print(
        "\nClassification report"
    )

    print("-" * 50)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Pass/Distinction",
                "Fail/Withdrawn",
            ],
            zero_division=0,
        )
    )

    return (
        metrics,
        predictions,
        probabilities,
    )


# ---------------------------------------------------------
# Extract Logistic Regression coefficients
# ---------------------------------------------------------

def extract_coefficients(
    model,
):
    """
    Retrieve transformed feature names and Logistic
    Regression coefficients.

    Positive coefficient:
        increases predicted risk

    Negative coefficient:
        decreases predicted risk
    """

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    coefficients = (
        classifier
        .coef_[0]
    )

    coefficient_table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(
                coefficients
            ),
        }
    )

    coefficient_table = (
        coefficient_table
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return coefficient_table


# ---------------------------------------------------------
# Save outputs
# ---------------------------------------------------------

def save_outputs(
    model,
    metrics,
    coefficient_table,
    test_results,
):
    """
    Save fitted model and experiment results.
    """

    model_file = (
        MODEL_DIR
        / "logistic_regression_day60.joblib"
    )

    metrics_file = (
        RESULTS_DIR
        / "logistic_regression_metrics.csv"
    )

    coefficients_file = (
        RESULTS_DIR
        / "logistic_regression_coefficients.csv"
    )

    predictions_file = (
        RESULTS_DIR
        / "logistic_regression_test_predictions.csv"
    )

    joblib.dump(
        model,
        model_file,
    )

    pd.DataFrame(
        [
            metrics
        ]
    ).to_csv(
        metrics_file,
        index=False,
    )

    coefficient_table.to_csv(
        coefficients_file,
        index=False,
    )

    test_results.to_csv(
        predictions_file,
        index=False,
    )

    print(
        "\nSaved model:"
    )

    print(
        model_file
    )

    print(
        "\nSaved metrics:"
    )

    print(
        metrics_file
    )

    print(
        "\nSaved coefficients:"
    )

    print(
        coefficients_file
    )

    print(
        "\nSaved test predictions:"
    )

    print(
        predictions_file
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "\nTraining Day-60 Logistic Regression baseline..."
    )

    # -----------------------------------------------------
    # 1. Load learner profiles
    # -----------------------------------------------------

    profiles = load_profiles()

    print(
        f"\nLoaded learner profiles: "
        f"{len(profiles):,}"
    )

    # -----------------------------------------------------
    # 2. Prepare model data
    # -----------------------------------------------------

    (
        X,
        y,
        learner_ids,
        model_features,
    ) = prepare_features(
        profiles
    )

    print(
        f"Model features: "
        f"{len(model_features)}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        y
        .value_counts()
        .sort_index()
    )

    # -----------------------------------------------------
    # 3. Train/test split
    #
    # Stratification preserves class proportions.
    # -----------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        learner_ids_train,
        learner_ids_test,
    ) = train_test_split(
        X,
        y,
        learner_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"\nTraining learners: "
        f"{len(X_train):,}"
    )

    print(
        f"Test learners: "
        f"{len(X_test):,}"
    )

    # -----------------------------------------------------
    # 4. Build model pipeline
    # -----------------------------------------------------

    model = build_model_pipeline(
        model_features
    )

    # -----------------------------------------------------
    # 5. Train
    # -----------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    print(
        "\nModel training complete."
    )

    # -----------------------------------------------------
    # 6. Evaluate
    # -----------------------------------------------------

    (
        metrics,
        predictions,
        probabilities,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # 7. Prepare prediction output
    # -----------------------------------------------------

    test_results = pd.DataFrame(
        {
            "id_student":
                learner_ids_test
                .values,

            "actual_risk_target":
                y_test
                .values,

            "predicted_risk_target":
                predictions,

            "predicted_risk_probability":
                probabilities,
        }
    )

    # -----------------------------------------------------
    # 8. Extract coefficients
    # -----------------------------------------------------

    coefficient_table = (
        extract_coefficients(
            model
        )
    )

    print(
        "\nTop Logistic Regression coefficients"
    )

    print("-" * 70)

    print(
        coefficient_table[
            [
                "feature",
                "coefficient",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 9. Save model and results
    # -----------------------------------------------------

    save_outputs(
        model=model,
        metrics=metrics,
        coefficient_table=coefficient_table,
        test_results=test_results,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BASELINE MODEL COMPLETE"
    )

    print(
        "=" * 70
    )