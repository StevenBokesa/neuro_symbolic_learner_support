from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------
# Load OULAD
# ---------------------------------------------------------

def load_oulad():
    """
    Load the seven core OULAD CSV tables.
    """

    tables = {
        "student_info": pd.read_csv(
            RAW_DATA_DIR / "studentInfo.csv"
        ),

        "student_registration": pd.read_csv(
            RAW_DATA_DIR / "studentRegistration.csv"
        ),

        "student_assessment": pd.read_csv(
            RAW_DATA_DIR / "studentAssessment.csv"
        ),

        "assessments": pd.read_csv(
            RAW_DATA_DIR / "assessments.csv"
        ),

        "student_vle": pd.read_csv(
            RAW_DATA_DIR / "studentVle.csv"
        ),

        "vle": pd.read_csv(
            RAW_DATA_DIR / "vle.csv"
        ),

        "courses": pd.read_csv(
            RAW_DATA_DIR / "courses.csv"
        ),
    }

    return tables


# ---------------------------------------------------------
# Filter to one module and presentation
# ---------------------------------------------------------

def filter_module_presentation(
    tables,
    code_module,
    code_presentation,
):
    """
    Scope all OULAD tables to one module presentation.

    Most OULAD tables contain code_module and
    code_presentation directly.

    studentAssessment does not contain those columns, so
    it must be scoped indirectly through id_assessment.
    """

    filtered = {}

    # -----------------------------------------------------
    # First filter tables that directly contain the module
    # and presentation identifiers.
    # -----------------------------------------------------

    for name, df in tables.items():

        if (
            "code_module" in df.columns
            and "code_presentation" in df.columns
        ):

            filtered[name] = df[
                (df["code_module"] == code_module)
                & (
                    df["code_presentation"]
                    == code_presentation
                )
            ].copy()

        else:
            filtered[name] = df.copy()

    # -----------------------------------------------------
    # Scope studentAssessment through assessments.
    #
    # Relationship:
    #
    # studentAssessment.id_assessment
    #              ↓
    # assessments.id_assessment
    #              ↓
    # code_module + code_presentation
    # -----------------------------------------------------

    scoped_assessment_ids = (
        filtered["assessments"][
            "id_assessment"
        ]
        .unique()
    )

    filtered["student_assessment"] = (
        tables["student_assessment"][
            tables["student_assessment"][
                "id_assessment"
            ].isin(scoped_assessment_ids)
        ]
        .copy()
    )

    # -----------------------------------------------------
    # Extra safety check:
    #
    # Only keep assessment submissions belonging to
    # learners enrolled in this module presentation.
    # -----------------------------------------------------

    scoped_learner_ids = (
        filtered["student_info"][
            "id_student"
        ]
        .unique()
    )

    filtered["student_assessment"] = (
        filtered["student_assessment"][
            filtered["student_assessment"][
                "id_student"
            ].isin(scoped_learner_ids)
        ]
        .copy()
    )

    return filtered


# ---------------------------------------------------------
# Apply temporal cut-off
# ---------------------------------------------------------

def filter_to_cutoff_day(
    tables,
    cutoff_day=60,
):
    """
    Restrict time-dependent records to evidence available
    on or before the selected course day.

    This is essential to prevent future information from
    leaking into an earlier learner-risk prediction.
    """

    filtered = {
        name: df.copy()
        for name, df in tables.items()
    }

    # -----------------------------------------------------
    # VLE interactions
    #
    # studentVle.date represents the course-relative day.
    #
    # Keep activity occurring on or before Day 60.
    # Negative dates are retained because they represent
    # activity occurring before the formal course start and
    # were therefore available before the prediction point.
    # -----------------------------------------------------

    if "date" in filtered["student_vle"].columns:

        filtered["student_vle"] = (
            filtered["student_vle"][
                filtered["student_vle"][
                    "date"
                ] <= cutoff_day
            ]
            .copy()
        )

    # -----------------------------------------------------
    # Student assessment submissions
    #
    # Keep only submissions actually made on or before
    # the observation point.
    # -----------------------------------------------------

    if (
        "date_submitted"
        in filtered["student_assessment"].columns
    ):

        filtered["student_assessment"] = (
            filtered["student_assessment"][
                filtered[
                    "student_assessment"
                ]["date_submitted"]
                <= cutoff_day
            ]
            .copy()
        )

    # -----------------------------------------------------
    # Assessment definitions
    #
    # Keep assessments whose scheduled due date is on or
    # before the observation point.
    #
    # Assessments with missing dates are excluded from the
    # Day-60 assessment set for now.
    # -----------------------------------------------------

    if "date" in filtered["assessments"].columns:

        assessments = filtered["assessments"]

        filtered["assessments"] = (
            assessments[
                assessments["date"].notna()
                & (
                    assessments["date"]
                    <= cutoff_day
                )
            ]
            .copy()
        )

    return filtered


# ---------------------------------------------------------
# Display dataset information
# ---------------------------------------------------------

def print_table_shapes(
    tables,
    title,
):
    """
    Print table dimensions.
    """

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for name, df in tables.items():

        print(
            f"{name:<25} "
            f"rows={df.shape[0]:>10,} "
            f"columns={df.shape[1]}"
        )


# ---------------------------------------------------------
# Display learner outcome distribution
# ---------------------------------------------------------

def print_outcome_distribution(
    tables,
):
    """
    Print the final-result distribution.

    final_result represents future outcome information.

    It will later be transformed into the prediction target:

        1 = Fail or Withdrawn
        0 = Pass or Distinction

    It must not be used as an input feature.
    """

    student_info = tables[
        "student_info"
    ]

    if (
        "final_result"
        not in student_info.columns
    ):

        print(
            "\nNo final_result column found."
        )

        return

    print(
        "\nFinal result distribution"
    )

    print("-" * 50)

    distribution = (
        student_info[
            "final_result"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        distribution
    )


# ---------------------------------------------------------
# Display binary target distribution
# ---------------------------------------------------------

def print_binary_target_distribution(
    tables,
):
    """
    Show the planned binary prediction target.

    At risk:
        Fail or Withdrawn

    Successful:
        Pass or Distinction
    """

    student_info = (
        tables["student_info"]
        .copy()
    )

    student_info[
        "risk_target"
    ] = (
        student_info[
            "final_result"
        ]
        .isin(
            [
                "Fail",
                "Withdrawn",
            ]
        )
        .astype(int)
    )

    print(
        "\nPlanned binary target"
    )

    print("-" * 50)

    counts = (
        student_info[
            "risk_target"
        ]
        .value_counts()
        .sort_index()
    )

    successful = counts.get(
        0,
        0,
    )

    at_risk = counts.get(
        1,
        0,
    )

    print(
        f"0 = Pass/Distinction: "
        f"{successful:,}"
    )

    print(
        f"1 = Fail/Withdrawn:   "
        f"{at_risk:,}"
    )


# ---------------------------------------------------------
# Temporal diagnostics
# ---------------------------------------------------------

def print_temporal_diagnostics(
    scoped_tables,
    cutoff_tables,
    cutoff_day,
):
    """
    Show how the temporal filtering changed the main
    event-level datasets.
    """

    print(
        "\nTemporal filtering diagnostics"
    )

    print("-" * 50)

    # -----------------------------------------------------
    # VLE counts
    # -----------------------------------------------------

    before_vle = len(
        scoped_tables[
            "student_vle"
        ]
    )

    after_vle = len(
        cutoff_tables[
            "student_vle"
        ]
    )

    print(
        f"VLE records before cut-off: "
        f"{before_vle:,}"
    )

    print(
        f"VLE records by Day "
        f"{cutoff_day}: "
        f"{after_vle:,}"
    )

    # -----------------------------------------------------
    # Assessment submission counts
    # -----------------------------------------------------

    before_assessment = len(
        scoped_tables[
            "student_assessment"
        ]
    )

    after_assessment = len(
        cutoff_tables[
            "student_assessment"
        ]
    )

    print()

    print(
        f"Assessment submissions "
        f"before cut-off: "
        f"{before_assessment:,}"
    )

    print(
        f"Assessment submissions "
        f"by Day {cutoff_day}: "
        f"{after_assessment:,}"
    )

    # -----------------------------------------------------
    # VLE date range
    # -----------------------------------------------------

    if after_vle > 0:

        min_vle_date = (
            cutoff_tables[
                "student_vle"
            ]["date"]
            .min()
        )

        max_vle_date = (
            cutoff_tables[
                "student_vle"
            ]["date"]
            .max()
        )

        print()

        print(
            f"Observed VLE date range: "
            f"{min_vle_date} "
            f"to {max_vle_date}"
        )

    # -----------------------------------------------------
    # Assessment submission range
    # -----------------------------------------------------

    if after_assessment > 0:

        min_submission_date = (
            cutoff_tables[
                "student_assessment"
            ][
                "date_submitted"
            ]
            .min()
        )

        max_submission_date = (
            cutoff_tables[
                "student_assessment"
            ][
                "date_submitted"
            ]
            .max()
        )

        print(
            f"Observed assessment "
            f"submission range: "
            f"{min_submission_date} "
            f"to {max_submission_date}"
        )


# ---------------------------------------------------------
# Assessment scoping diagnostics
# ---------------------------------------------------------

def print_assessment_diagnostics(
    scoped_tables,
):
    """
    Confirm that studentAssessment has been correctly
    restricted to assessments belonging to the selected
    module presentation.
    """

    assessments = (
        scoped_tables[
            "assessments"
        ]
    )

    student_assessment = (
        scoped_tables[
            "student_assessment"
        ]
    )

    valid_assessment_ids = set(
        assessments[
            "id_assessment"
        ]
        .unique()
    )

    submission_assessment_ids = set(
        student_assessment[
            "id_assessment"
        ]
        .unique()
    )

    print(
        "\nAssessment scoping diagnostics"
    )

    print("-" * 50)

    print(
        f"Assessments in selected "
        f"presentation: "
        f"{len(valid_assessment_ids)}"
    )

    print(
        f"Assessment IDs appearing "
        f"in student submissions: "
        f"{len(submission_assessment_ids)}"
    )

    invalid_ids = (
        submission_assessment_ids
        - valid_assessment_ids
    )

    if not invalid_ids:

        print(
            "Assessment scoping check: PASS"
        )

    else:

        print(
            "Assessment scoping check: FAIL"
        )

        print(
            f"Unexpected assessment IDs: "
            f"{sorted(invalid_ids)}"
        )


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------
    # 1. Experiment configuration
    # -----------------------------------------------------

    CODE_MODULE = "BBB"

    CODE_PRESENTATION = "2014J"

    CUTOFF_DAY = 60

    # -----------------------------------------------------
    # 2. Load complete OULAD dataset
    # -----------------------------------------------------

    data = load_oulad()

    print_table_shapes(
        data,
        "FULL OULAD DATASET",
    )

    # -----------------------------------------------------
    # 3. Scope to one module presentation
    # -----------------------------------------------------

    scoped = (
        filter_module_presentation(
            data,
            code_module=CODE_MODULE,
            code_presentation=(
                CODE_PRESENTATION
            ),
        )
    )

    print_table_shapes(
        scoped,
        (
            "SCOPED EXPERIMENT: "
            f"{CODE_MODULE} "
            f"{CODE_PRESENTATION}"
        ),
    )

    # -----------------------------------------------------
    # 4. Verify assessment scoping
    # -----------------------------------------------------

    print_assessment_diagnostics(
        scoped
    )

    # -----------------------------------------------------
    # 5. Apply temporal boundary
    # -----------------------------------------------------

    cutoff_data = (
        filter_to_cutoff_day(
            scoped,
            cutoff_day=CUTOFF_DAY,
        )
    )

    print_table_shapes(
        cutoff_data,
        (
            "INFORMATION AVAILABLE "
            f"BY DAY {CUTOFF_DAY}"
        ),
    )

    # -----------------------------------------------------
    # 6. Show final learner outcomes
    #
    # These are retained only to construct the prediction
    # target later.
    # -----------------------------------------------------

    print_outcome_distribution(
        scoped
    )

    # -----------------------------------------------------
    # 7. Show binary target
    # -----------------------------------------------------

    print_binary_target_distribution(
        scoped
    )

    # -----------------------------------------------------
    # 8. Count learners
    # -----------------------------------------------------

    number_of_students = (
        scoped[
            "student_info"
        ][
            "id_student"
        ]
        .nunique()
    )

    print(
        "\nLearners in experiment"
    )

    print("-" * 50)

    print(
        f"Unique learners: "
        f"{number_of_students:,}"
    )

    # -----------------------------------------------------
    # 9. Temporal diagnostics
    # -----------------------------------------------------

    print_temporal_diagnostics(
        scoped_tables=scoped,
        cutoff_tables=cutoff_data,
        cutoff_day=CUTOFF_DAY,
    )

    # -----------------------------------------------------
    # 10. Final confirmation
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Experiment preparation complete."
    )

    print(
        f"Prediction point: "
        f"Day {CUTOFF_DAY}"
    )

    print(
        "Module presentation: "
        f"{CODE_MODULE} "
        f"{CODE_PRESENTATION}"
    )

    print(
        "=" * 70
    )