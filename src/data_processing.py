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
    Filter OULAD tables to one module and presentation
    where those columns are available.
    """

    filtered = {}

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

    return filtered


# ---------------------------------------------------------
# Apply temporal cut-off
# ---------------------------------------------------------

def filter_to_cutoff_day(
    tables,
    cutoff_day=60,
):
    """
    Restrict time-dependent records to information
    available on or before the selected course day.

    This prevents future information from leaking into
    an earlier learner-risk prediction.
    """

    filtered = {
        name: df.copy()
        for name, df in tables.items()
    }

    # -----------------------------------------------------
    # VLE activity
    #
    # studentVle.date is relative to the presentation start.
    # Keep only interactions occurring by the cut-off.
    # -----------------------------------------------------

    if "date" in filtered["student_vle"].columns:

        filtered["student_vle"] = (
            filtered["student_vle"][
                filtered["student_vle"]["date"]
                <= cutoff_day
            ]
            .copy()
        )

    # -----------------------------------------------------
    # Assessment submissions
    #
    # Keep submissions actually made by the learner
    # on or before the cut-off.
    # -----------------------------------------------------

    if (
        "date_submitted"
        in filtered["student_assessment"].columns
    ):

        filtered["student_assessment"] = (
            filtered["student_assessment"][
                filtered["student_assessment"][
                    "date_submitted"
                ]
                <= cutoff_day
            ]
            .copy()
        )

    # -----------------------------------------------------
    # Assessment definitions
    #
    # Keep assessments whose scheduled date is on or before
    # the cut-off.
    #
    # Missing dates are excluded from this temporal set.
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

def print_outcome_distribution(tables):
    """
    Print the final-result distribution.

    final_result must only be used as the prediction target,
    never as a Day-60 input feature.
    """

    student_info = tables["student_info"]

    if "final_result" not in student_info.columns:
        print("\nNo final_result column found.")
        return

    print("\nFinal result distribution")
    print("-" * 50)

    distribution = (
        student_info["final_result"]
        .value_counts(dropna=False)
    )

    print(distribution)


# ---------------------------------------------------------
# Print date diagnostics
# ---------------------------------------------------------

def print_temporal_diagnostics(
    scoped_tables,
    cutoff_tables,
    cutoff_day,
):
    """
    Show how the temporal cut-off changed the two main
    event-level datasets.
    """

    print("\nTemporal filtering diagnostics")
    print("-" * 50)

    before_vle = len(
        scoped_tables["student_vle"]
    )

    after_vle = len(
        cutoff_tables["student_vle"]
    )

    before_assessments = len(
        scoped_tables["student_assessment"]
    )

    after_assessments = len(
        cutoff_tables["student_assessment"]
    )

    print(
        f"VLE records before cut-off: "
        f"{before_vle:,}"
    )

    print(
        f"VLE records by Day {cutoff_day}: "
        f"{after_vle:,}"
    )

    print(
        f"\nAssessment submissions before cut-off: "
        f"{before_assessments:,}"
    )

    print(
        f"Assessment submissions by Day {cutoff_day}: "
        f"{after_assessments:,}"
    )

    if after_vle > 0:

        min_vle_date = (
            cutoff_tables["student_vle"]["date"]
            .min()
        )

        max_vle_date = (
            cutoff_tables["student_vle"]["date"]
            .max()
        )

        print(
            f"\nObserved VLE date range: "
            f"{min_vle_date} to {max_vle_date}"
        )

    if after_assessments > 0:

        min_submission_date = (
            cutoff_tables[
                "student_assessment"
            ]["date_submitted"]
            .min()
        )

        max_submission_date = (
            cutoff_tables[
                "student_assessment"
            ]["date_submitted"]
            .max()
        )

        print(
            f"Observed assessment submission range: "
            f"{min_submission_date} "
            f"to {max_submission_date}"
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
    # 3. Scope the experiment
    # -----------------------------------------------------

    scoped = filter_module_presentation(
        data,
        code_module=CODE_MODULE,
        code_presentation=CODE_PRESENTATION,
    )

    print_table_shapes(
        scoped,
        (
            f"SCOPED EXPERIMENT: "
            f"{CODE_MODULE} "
            f"{CODE_PRESENTATION}"
        ),
    )

    # -----------------------------------------------------
    # 4. Establish Day-60 temporal boundary
    # -----------------------------------------------------

    cutoff_data = filter_to_cutoff_day(
        scoped,
        cutoff_day=CUTOFF_DAY,
    )

    print_table_shapes(
        cutoff_data,
        (
            f"INFORMATION AVAILABLE "
            f"BY DAY {CUTOFF_DAY}"
        ),
    )

    # -----------------------------------------------------
    # 5. Show target distribution
    #
    # Important:
    # final_result is future outcome information.
    # It is retained for constructing the model target,
    # not as an input feature.
    # -----------------------------------------------------

    print_outcome_distribution(
        scoped
    )

    # -----------------------------------------------------
    # 6. Count learners
    # -----------------------------------------------------

    number_of_students = (
        scoped["student_info"][
            "id_student"
        ]
        .nunique()
    )

    print("\nLearners in experiment")
    print("-" * 50)

    print(
        f"Unique learners: "
        f"{number_of_students:,}"
    )

    # -----------------------------------------------------
    # 7. Temporal diagnostics
    # -----------------------------------------------------

    print_temporal_diagnostics(
        scoped_tables=scoped,
        cutoff_tables=cutoff_data,
        cutoff_day=CUTOFF_DAY,
    )

    # -----------------------------------------------------
    # 8. Confirmation
    # -----------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "Experiment preparation complete."
    )

    print(
        f"Prediction point: Day {CUTOFF_DAY}"
    )

    print(
        f"Module presentation: "
        f"{CODE_MODULE} "
        f"{CODE_PRESENTATION}"
    )

    print("=" * 70)