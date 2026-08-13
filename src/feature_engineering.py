from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing import (
    load_oulad,
    filter_module_presentation,
    filter_to_cutoff_day,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

CODE_MODULE = "BBB"
CODE_PRESENTATION = "2014J"
CUTOFF_DAY = 60

RECENT_WINDOW_DAYS = 14
PREVIOUS_WINDOW_DAYS = 14


# ---------------------------------------------------------
# Build prediction target
# ---------------------------------------------------------

def build_target(student_info):
    """
    Convert OULAD final_result into a binary prediction target.

    1 = Fail or Withdrawn
    0 = Pass or Distinction

    final_result is used only to construct the target and must
    not later be passed to the ML model as an input feature.
    """

    target = student_info[
        [
            "id_student",
            "final_result",
        ]
    ].copy()

    target["risk_target"] = (
        target["final_result"]
        .isin(
            [
                "Fail",
                "Withdrawn",
            ]
        )
        .astype(int)
    )

    return target


# ---------------------------------------------------------
# Assessment features
# ---------------------------------------------------------

def build_assessment_features(
    scoped_tables,
    cutoff_day,
):
    """
    Construct assessment features using assessments that were
    formally due on or before the observation day.

    Features:
        weighted_assessment_average
        latest_assessment_score
        assessment_trend
        assessments_submitted
        assessment_completion_rate
        late_submissions
    """

    assessments = (
        scoped_tables["assessments"]
        .copy()
    )

    student_assessment = (
        scoped_tables["student_assessment"]
        .copy()
    )

    # -----------------------------------------------------
    # 1. Find assessments formally due by Day 60
    # -----------------------------------------------------

    due_assessments = assessments[
        assessments["date"].notna()
        & (
            assessments["date"]
            <= cutoff_day
        )
    ].copy()

    due_assessment_ids = (
        due_assessments[
            "id_assessment"
        ]
        .unique()
    )

    number_due = len(
        due_assessment_ids
    )

    print(
        f"Assessments due by Day "
        f"{cutoff_day}: {number_due}"
    )

    # -----------------------------------------------------
    # 2. Keep submissions for those due assessments only
    # -----------------------------------------------------

    submissions = student_assessment[
        student_assessment[
            "id_assessment"
        ].isin(
            due_assessment_ids
        )
        & (
            student_assessment[
                "date_submitted"
            ]
            <= cutoff_day
        )
    ].copy()

    # -----------------------------------------------------
    # 3. Join assessment metadata
    # -----------------------------------------------------

    submissions = submissions.merge(
        due_assessments[
            [
                "id_assessment",
                "date",
                "weight",
                "assessment_type",
            ]
        ],
        on="id_assessment",
        how="left",
    )

    # -----------------------------------------------------
    # 4. Determine whether submission was late
    # -----------------------------------------------------

    submissions[
        "is_late"
    ] = (
        submissions[
            "date_submitted"
        ]
        > submissions[
            "date"
        ]
    ).astype(int)

    # -----------------------------------------------------
    # 5. Weighted score contribution
    # -----------------------------------------------------

    submissions[
        "weighted_score"
    ] = (
        submissions[
            "score"
        ]
        * submissions[
            "weight"
        ]
    )

    # -----------------------------------------------------
    # 6. Aggregate weighted assessment average
    # -----------------------------------------------------

    weighted = (
        submissions
        .groupby(
            "id_student"
        )
        .agg(
            weighted_score_sum=(
                "weighted_score",
                "sum",
            ),
            submitted_weight=(
                "weight",
                "sum",
            ),
            assessments_submitted=(
                "id_assessment",
                "nunique",
            ),
            late_submissions=(
                "is_late",
                "sum",
            ),
        )
        .reset_index()
    )

    weighted[
        "weighted_assessment_average"
    ] = np.where(
        weighted[
            "submitted_weight"
        ] > 0,
        weighted[
            "weighted_score_sum"
        ]
        / weighted[
            "submitted_weight"
        ],
        np.nan,
    )

    # -----------------------------------------------------
    # 7. Latest score
    # -----------------------------------------------------

    ordered = submissions.sort_values(
        [
            "id_student",
            "date_submitted",
            "id_assessment",
        ]
    )

    latest_score = (
        ordered
        .groupby(
            "id_student"
        )
        .tail(1)[
            [
                "id_student",
                "score",
            ]
        ]
        .rename(
            columns={
                "score":
                    "latest_assessment_score"
            }
        )
    )

    # -----------------------------------------------------
    # 8. Assessment trend
    #
    # Simple implementation:
    # latest score - previous score
    # -----------------------------------------------------

    def calculate_trend(group):
        group = group.sort_values(
            [
                "date_submitted",
                "id_assessment",
            ]
        )

        scores = (
            group["score"]
            .dropna()
            .tolist()
        )

        if len(scores) < 2:
            return np.nan

        return (
            scores[-1]
            - scores[-2]
        )

    trends = (
        submissions
        .groupby(
            "id_student"
        )
        .apply(
            calculate_trend,
            include_groups=False,
        )
        .reset_index(
            name="assessment_trend"
        )
    )

    # -----------------------------------------------------
    # 9. Completion rate
    # -----------------------------------------------------

    if number_due > 0:

        weighted[
            "assessment_completion_rate"
        ] = (
            weighted[
                "assessments_submitted"
            ]
            / number_due
        )

    else:

        weighted[
            "assessment_completion_rate"
        ] = np.nan

    # -----------------------------------------------------
    # 10. Combine assessment features
    # -----------------------------------------------------

    features = (
        weighted
        .merge(
            latest_score,
            on="id_student",
            how="outer",
        )
        .merge(
            trends,
            on="id_student",
            how="outer",
        )
    )

    columns = [
        "id_student",
        "weighted_assessment_average",
        "latest_assessment_score",
        "assessment_trend",
        "assessments_submitted",
        "assessment_completion_rate",
        "late_submissions",
    ]

    return features[
        columns
    ]


# ---------------------------------------------------------
# Engagement / VLE features
# ---------------------------------------------------------

def build_engagement_features(
    student_vle,
    cutoff_day,
    recent_window=14,
):
    """
    Construct VLE engagement features available by Day 60.

    Features:
        total_clicks
        recent_14_day_clicks
        previous_14_day_clicks
        engagement_change
        active_days
        days_since_last_activity
    """

    vle = student_vle.copy()

    # -----------------------------------------------------
    # Total clicks by observation point
    # -----------------------------------------------------

    total_clicks = (
        vle
        .groupby(
            "id_student"
        )["sum_click"]
        .sum()
        .reset_index(
            name="total_clicks"
        )
    )

    # -----------------------------------------------------
    # Active days
    # -----------------------------------------------------

    active_days = (
        vle
        .groupby(
            "id_student"
        )["date"]
        .nunique()
        .reset_index(
            name="active_days"
        )
    )

    # -----------------------------------------------------
    # Last activity date / inactivity
    # -----------------------------------------------------

    last_activity = (
        vle
        .groupby(
            "id_student"
        )["date"]
        .max()
        .reset_index(
            name="last_activity_day"
        )
    )

    last_activity[
        "days_since_last_activity"
    ] = (
        cutoff_day
        - last_activity[
            "last_activity_day"
        ]
    )

    # -----------------------------------------------------
    # Recent 14-day period
    #
    # For Day 60:
    # Day 47 through Day 60 inclusive
    # -----------------------------------------------------

    recent_start = (
        cutoff_day
        - recent_window
        + 1
    )

    recent = vle[
        (vle["date"] >= recent_start)
        & (
            vle["date"]
            <= cutoff_day
        )
    ]

    recent_clicks = (
        recent
        .groupby(
            "id_student"
        )["sum_click"]
        .sum()
        .reset_index(
            name="recent_14_day_clicks"
        )
    )

    # -----------------------------------------------------
    # Previous 14-day period
    #
    # For Day 60:
    # Day 33 through Day 46
    # -----------------------------------------------------

    previous_end = (
        recent_start
        - 1
    )

    previous_start = (
        previous_end
        - recent_window
        + 1
    )

    previous = vle[
        (
            vle["date"]
            >= previous_start
        )
        & (
            vle["date"]
            <= previous_end
        )
    ]

    previous_clicks = (
        previous
        .groupby(
            "id_student"
        )["sum_click"]
        .sum()
        .reset_index(
            name="previous_14_day_clicks"
        )
    )

    # -----------------------------------------------------
    # Merge engagement metrics
    # -----------------------------------------------------

    engagement = (
        total_clicks
        .merge(
            recent_clicks,
            on="id_student",
            how="outer",
        )
        .merge(
            previous_clicks,
            on="id_student",
            how="outer",
        )
        .merge(
            active_days,
            on="id_student",
            how="outer",
        )
        .merge(
            last_activity,
            on="id_student",
            how="outer",
        )
    )

    # -----------------------------------------------------
    # Missing click counts mean no activity in window
    # -----------------------------------------------------

    click_columns = [
        "total_clicks",
        "recent_14_day_clicks",
        "previous_14_day_clicks",
    ]

    engagement[
        click_columns
    ] = (
        engagement[
            click_columns
        ]
        .fillna(0)
    )

    # -----------------------------------------------------
    # Engagement change
    #
    # (recent - previous) / max(previous, 1)
    #
    # Example:
    # previous = 60
    # recent = 21
    #
    # (21 - 60) / 60 = -0.65
    # -----------------------------------------------------

    engagement[
        "engagement_change"
    ] = (
        engagement[
            "recent_14_day_clicks"
        ]
        - engagement[
            "previous_14_day_clicks"
        ]
    ) / (
        engagement[
            "previous_14_day_clicks"
        ]
        .clip(
            lower=1
        )
    )

    return engagement


# ---------------------------------------------------------
# Context features
# ---------------------------------------------------------

def build_context_features(
    student_info,
):
    """
    Construct simple learner-context features.

    For the first model we retain:
        previous_attempts
        studied_credits

    Other demographic variables can later be retained for
    subgroup/fairness analysis rather than direct intervention
    rules.
    """

    return (
        student_info[
            [
                "id_student",
                "num_of_prev_attempts",
                "studied_credits",
            ]
        ]
        .rename(
            columns={
                "num_of_prev_attempts":
                    "previous_attempts"
            }
        )
        .copy()
    )


# ---------------------------------------------------------
# Build complete learner profile
# ---------------------------------------------------------

def build_learner_profiles(
    scoped_tables,
    cutoff_tables,
    cutoff_day,
):
    """
    Combine assessment, engagement, context and target data
    into one row per learner.
    """

    student_info = (
        scoped_tables[
            "student_info"
        ]
        .copy()
    )

    # -----------------------------------------------------
    # Start with every learner in the selected presentation
    # -----------------------------------------------------

    learner_profiles = (
        student_info[
            [
                "id_student",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    # -----------------------------------------------------
    # Assessment features
    # -----------------------------------------------------

    assessment_features = (
        build_assessment_features(
            scoped_tables,
            cutoff_day=cutoff_day,
        )
    )

    # -----------------------------------------------------
    # Engagement features
    #
    # cutoff_tables["student_vle"] already contains activity
    # only up to Day 60.
    # -----------------------------------------------------

    engagement_features = (
        build_engagement_features(
            cutoff_tables[
                "student_vle"
            ],
            cutoff_day=cutoff_day,
        )
    )

    # -----------------------------------------------------
    # Context
    # -----------------------------------------------------

    context_features = (
        build_context_features(
            student_info
        )
    )

    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    target = build_target(
        student_info
    )

    # -----------------------------------------------------
    # Merge everything
    # -----------------------------------------------------

    learner_profiles = (
        learner_profiles
        .merge(
            assessment_features,
            on="id_student",
            how="left",
        )
        .merge(
            engagement_features,
            on="id_student",
            how="left",
        )
        .merge(
            context_features,
            on="id_student",
            how="left",
        )
        .merge(
            target,
            on="id_student",
            how="left",
        )
    )

    # -----------------------------------------------------
    # Learners with no VLE records
    # -----------------------------------------------------

    zero_fill_columns = [
        "total_clicks",
        "recent_14_day_clicks",
        "previous_14_day_clicks",
        "active_days",
    ]

    learner_profiles[
        zero_fill_columns
    ] = (
        learner_profiles[
            zero_fill_columns
        ]
        .fillna(0)
    )

    # -----------------------------------------------------
    # If learner has no activity at all before Day 60,
    # interpret inactivity as the entire observation period.
    #
    # Because activity before course day 0 is possible, we
    # leave last_activity_day missing but assign inactivity
    # conservatively to cutoff_day.
    # -----------------------------------------------------

    learner_profiles[
        "days_since_last_activity"
    ] = (
        learner_profiles[
            "days_since_last_activity"
        ]
        .fillna(
            cutoff_day
        )
    )

    # -----------------------------------------------------
    # Missing assessment participation
    #
    # If assessments were due but the student submitted none:
    # submitted count = 0
    # completion = 0
    # late submissions = 0
    #
    # Scores/trend remain NaN because there is no observed
    # score and we must not invent one.
    # -----------------------------------------------------

    learner_profiles[
        "assessments_submitted"
    ] = (
        learner_profiles[
            "assessments_submitted"
        ]
        .fillna(0)
    )

    learner_profiles[
        "assessment_completion_rate"
    ] = (
        learner_profiles[
            "assessment_completion_rate"
        ]
        .fillna(0)
    )

    learner_profiles[
        "late_submissions"
    ] = (
        learner_profiles[
            "late_submissions"
        ]
        .fillna(0)
    )

    return learner_profiles


# ---------------------------------------------------------
# Validate generated profiles
# ---------------------------------------------------------

def validate_profiles(
    profiles,
    expected_students,
):
    """
    Basic quality checks before saving the modelling dataset.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LEARNER PROFILE VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Expected learners: "
        f"{expected_students:,}"
    )

    print(
        f"Generated profiles: "
        f"{len(profiles):,}"
    )

    duplicate_students = (
        profiles[
            "id_student"
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate learner rows: "
        f"{duplicate_students}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        profiles[
            "risk_target"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nMissing values:"
    )

    print(
        profiles
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nFeature summary:"
    )

    numeric_columns = [
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

    print(
        profiles[
            numeric_columns
        ]
        .describe()
        .transpose()
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "\nBuilding Day-60 learner profiles..."
    )

    # -----------------------------------------------------
    # 1. Load OULAD
    # -----------------------------------------------------

    data = load_oulad()

    # -----------------------------------------------------
    # 2. Scope to BBB 2014J
    # -----------------------------------------------------

    scoped = (
        filter_module_presentation(
            data,
            code_module=CODE_MODULE,
            code_presentation=CODE_PRESENTATION,
        )
    )

    # -----------------------------------------------------
    # 3. Apply Day-60 boundary
    # -----------------------------------------------------

    cutoff_tables = (
        filter_to_cutoff_day(
            scoped,
            cutoff_day=CUTOFF_DAY,
        )
    )

    # -----------------------------------------------------
    # 4. Build learner profiles
    # -----------------------------------------------------

    profiles = (
        build_learner_profiles(
            scoped_tables=scoped,
            cutoff_tables=cutoff_tables,
            cutoff_day=CUTOFF_DAY,
        )
    )

    # -----------------------------------------------------
    # 5. Validate
    # -----------------------------------------------------

    expected_students = (
        scoped[
            "student_info"
        ][
            "id_student"
        ]
        .nunique()
    )

    validate_profiles(
        profiles,
        expected_students=expected_students,
    )

    # -----------------------------------------------------
    # 6. Save output
    # -----------------------------------------------------

    output_file = (
        PROCESSED_DATA_DIR
        / (
            f"learner_profiles_"
            f"{CODE_MODULE}_"
            f"{CODE_PRESENTATION}_"
            f"day{CUTOFF_DAY}.csv"
        )
    )

    profiles.to_csv(
        output_file,
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FEATURE ENGINEERING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Saved learner profiles to:"
    )

    print(
        output_file
    )

    print(
        f"\nRows: {len(profiles):,}"
    )

    print(
        f"Columns: "
        f"{len(profiles.columns)}"
    )