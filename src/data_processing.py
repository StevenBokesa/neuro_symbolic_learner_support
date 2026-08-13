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
    Filter all OULAD tables that contain both code_module
    and code_presentation.

    Example:
        code_module="BBB"
        code_presentation="2014J"
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
# Display dataset information
# ---------------------------------------------------------

def print_table_shapes(tables, title):
    """
    Print table names and their number of rows and columns.
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for name, df in tables.items():
        print(f"{name:<25} {df.shape}")


# ---------------------------------------------------------
# Display learner outcome distribution
# ---------------------------------------------------------

def print_outcome_distribution(tables):
    """
    Display the final_result distribution for the selected
    module presentation.
    """

    student_info = tables["student_info"]

    if "final_result" not in student_info.columns:
        print("\nNo final_result column found.")
        return

    print("\nFinal result distribution:")
    print("-" * 40)

    print(
        student_info["final_result"]
        .value_counts(dropna=False)
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    # ---------------------------------------------
    # 1. Load full OULAD dataset
    # ---------------------------------------------

    data = load_oulad()

    print_table_shapes(
        data,
        "FULL OULAD DATASET",
    )

    # ---------------------------------------------
    # 2. Select one module presentation
    # ---------------------------------------------

    CODE_MODULE = "BBB"
    CODE_PRESENTATION = "2014J"

    scoped = filter_module_presentation(
        data,
        code_module=CODE_MODULE,
        code_presentation=CODE_PRESENTATION,
    )

    # ---------------------------------------------
    # 3. Display scoped dataset
    # ---------------------------------------------

    print_table_shapes(
        scoped,
        f"SCOPED EXPERIMENT: "
        f"{CODE_MODULE} {CODE_PRESENTATION}",
    )

    # ---------------------------------------------
    # 4. Display learner outcomes
    # ---------------------------------------------

    print_outcome_distribution(scoped)

    # ---------------------------------------------
    # 5. Basic learner count
    # ---------------------------------------------

    student_info = scoped["student_info"]

    number_of_students = (
        student_info["id_student"].nunique()
    )

    print("\nNumber of unique learners:")
    print(number_of_students)

    # ---------------------------------------------
    # 6. Confirmation
    # ---------------------------------------------

    print("\nData loading and experiment scoping complete.")