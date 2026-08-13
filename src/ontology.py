from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learner_profiles_BBB_2014J_day60.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learner_semantic_states_BBB_2014J_day60.csv"
)


# ---------------------------------------------------------
# Performance state
# ---------------------------------------------------------

def classify_performance(weighted_average):
    """
    Map weighted assessment average to an interpretable
    performance concept.
    """

    if pd.isna(weighted_average):
        return "InsufficientAssessmentEvidence"

    if weighted_average >= 70:
        return "StrongPerformance"

    if weighted_average >= 50:
        return "SatisfactoryPerformance"

    if weighted_average >= 40:
        return "BorderlinePerformance"

    return "LowPerformance"


# ---------------------------------------------------------
# Assessment trend state
# ---------------------------------------------------------

def classify_assessment_trend(assessment_trend):
    """
    Convert assessment change into a symbolic state.

    Thresholds:
        <= -10        SharplyDecliningAssessment
        -10 to -3     DecliningAssessment
        -3 to 3       StableAssessment
        3 to 10       ImprovingAssessment
        > 10          StronglyImprovingAssessment
    """

    if pd.isna(assessment_trend):
        return "InsufficientTrendEvidence"

    if assessment_trend <= -10:
        return "SharplyDecliningAssessment"

    if assessment_trend < -3:
        return "DecliningAssessment"

    if assessment_trend <= 3:
        return "StableAssessment"

    if assessment_trend <= 10:
        return "ImprovingAssessment"

    return "StronglyImprovingAssessment"


# ---------------------------------------------------------
# Engagement state
# ---------------------------------------------------------

def classify_engagement(
    engagement_change,
    recent_clicks,
    previous_clicks,
):
    """
    Convert VLE activity change into an interpretable
    engagement state.

    engagement_change is based on:

        (recent clicks - previous clicks)
        / max(previous clicks, 1)
    """

    if recent_clicks == 0 and previous_clicks == 0:
        return "NoObservedEngagement"

    if pd.isna(engagement_change):
        return "InsufficientEngagementEvidence"

    if engagement_change <= -0.50:
        return "SharplyDecliningEngagement"

    if engagement_change <= -0.20:
        return "DecliningEngagement"

    if engagement_change < 0.20:
        return "StableEngagement"

    if engagement_change < 0.50:
        return "IncreasingEngagement"

    return "StronglyIncreasingEngagement"


# ---------------------------------------------------------
# Inactivity state
# ---------------------------------------------------------

def classify_inactivity(days_since_last_activity):
    """
    Convert inactivity duration to an interpretable state.
    """

    if pd.isna(days_since_last_activity):
        return "UnknownInactivity"

    if days_since_last_activity >= 14:
        return "HighlyInactive"

    if days_since_last_activity >= 7:
        return "Inactive"

    if days_since_last_activity >= 3:
        return "RecentlyInactive"

    return "Active"


# ---------------------------------------------------------
# Assessment completion state
# ---------------------------------------------------------

def classify_completion(completion_rate):
    """
    Convert assessment completion rate into a semantic state.
    """

    if pd.isna(completion_rate):
        return "UnknownCompletion"

    if completion_rate >= 1.0:
        return "Complete"

    if completion_rate >= 0.50:
        return "PartialCompletion"

    if completion_rate > 0:
        return "LowCompletion"

    return "NoAssessmentSubmission"


# ---------------------------------------------------------
# Submission lateness state
# ---------------------------------------------------------

def classify_submission_state(late_submissions):
    """
    Represent assessment submission lateness.
    """

    if pd.isna(late_submissions):
        return "UnknownSubmissionTimeliness"

    if late_submissions == 0:
        return "NoLateSubmissions"

    if late_submissions == 1:
        return "LateSubmission"

    return "MultipleLateSubmissions"


# ---------------------------------------------------------
# Evidence sufficiency
# ---------------------------------------------------------

def classify_evidence_sufficiency(
    weighted_average,
    total_clicks,
    assessment_trend,
):
    """
    Distinguish sufficient evidence from partial or missing
    evidence.

    This prevents missing data from automatically being
    interpreted as high risk.
    """

    has_assessment = not pd.isna(weighted_average)

    has_engagement = (
        not pd.isna(total_clicks)
        and total_clicks > 0
    )

    has_trend = not pd.isna(assessment_trend)

    evidence_count = sum(
        [
            has_assessment,
            has_engagement,
            has_trend,
        ]
    )

    if evidence_count == 3:
        return "SufficientEvidence"

    if evidence_count >= 1:
        return "PartialEvidence"

    return "InsufficientEvidence"


# ---------------------------------------------------------
# Build semantic states
# ---------------------------------------------------------

def build_semantic_states(profiles):
    """
    Convert numerical learner evidence into interpretable
    semantic states.

    These states will later become RDF concepts in the
    knowledge graph.
    """

    df = profiles.copy()

    df["performance_state"] = (
        df["weighted_assessment_average"]
        .apply(classify_performance)
    )

    df["assessment_trend_state"] = (
        df["assessment_trend"]
        .apply(classify_assessment_trend)
    )

    df["engagement_state"] = df.apply(
        lambda row: classify_engagement(
            row["engagement_change"],
            row["recent_14_day_clicks"],
            row["previous_14_day_clicks"],
        ),
        axis=1,
    )

    df["inactivity_state"] = (
        df["days_since_last_activity"]
        .apply(classify_inactivity)
    )

    df["completion_state"] = (
        df["assessment_completion_rate"]
        .apply(classify_completion)
    )

    df["submission_state"] = (
        df["late_submissions"]
        .apply(classify_submission_state)
    )

    df["evidence_sufficiency"] = df.apply(
        lambda row: classify_evidence_sufficiency(
            row["weighted_assessment_average"],
            row["total_clicks"],
            row["assessment_trend"],
        ),
        axis=1,
    )

    return df


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_semantic_states(df):
    """
    Print distributions for the generated semantic states.
    """

    state_columns = [
        "performance_state",
        "assessment_trend_state",
        "engagement_state",
        "inactivity_state",
        "completion_state",
        "submission_state",
        "evidence_sufficiency",
    ]

    print("\n" + "=" * 70)
    print("SEMANTIC STATE VALIDATION")
    print("=" * 70)

    print(
        f"Learners represented: "
        f"{df['id_student'].nunique():,}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Duplicate learner rows: "
        f"{df['id_student'].duplicated().sum()}"
    )

    for column in state_columns:

        print("\n" + column)
        print("-" * 50)

        print(
            df[column]
            .value_counts(
                dropna=False
            )
        )


# ---------------------------------------------------------
# Display example learners
# ---------------------------------------------------------

def print_examples(df, n=10):
    """
    Print several examples showing the transformation from
    numerical evidence to semantic concepts.
    """

    columns = [
        "id_student",
        "weighted_assessment_average",
        "performance_state",
        "assessment_trend",
        "assessment_trend_state",
        "recent_14_day_clicks",
        "previous_14_day_clicks",
        "engagement_change",
        "engagement_state",
        "days_since_last_activity",
        "inactivity_state",
        "assessment_completion_rate",
        "completion_state",
        "late_submissions",
        "submission_state",
        "evidence_sufficiency",
        "risk_target",
    ]

    print("\n" + "=" * 70)
    print("EXAMPLE SEMANTIC LEARNER STATES")
    print("=" * 70)

    print(
        df[columns]
        .head(n)
        .to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "\nBuilding semantic learner evidence states..."
    )

    # -----------------------------------------------------
    # 1. Check learner-profile dataset exists
    # -----------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Learner profile dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run:\n"
            f"python -m src.feature_engineering"
        )

    # -----------------------------------------------------
    # 2. Load learner profiles
    # -----------------------------------------------------

    profiles = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Loaded learner profiles: "
        f"{len(profiles):,}"
    )

    # -----------------------------------------------------
    # 3. Generate semantic states
    # -----------------------------------------------------

    semantic_states = (
        build_semantic_states(
            profiles
        )
    )

    # -----------------------------------------------------
    # 4. Validate
    # -----------------------------------------------------

    validate_semantic_states(
        semantic_states
    )

    print_examples(
        semantic_states,
        n=10,
    )

    # -----------------------------------------------------
    # 5. Save
    # -----------------------------------------------------

    semantic_states.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SEMANTIC STATE GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nRows: "
        f"{len(semantic_states):,}"
    )

    print(
        f"Columns: "
        f"{len(semantic_states.columns)}"
    )