from pathlib import Path

import joblib
import pandas as pd
from rdflib import Graph, Literal, Namespace, XSD

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

RULE_RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learner_rule_results_BBB_2014J_day60.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "logistic_regression_day60.joblib"
)

REASONED_GRAPH_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "graphs"
    / "learner_graph_reasoned_BBB_2014J_day60.ttl"
)

INTEGRATED_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learner_integrated_decisions_BBB_2014J_day60.csv"
)

INTEGRATED_GRAPH_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "graphs"
    / "learner_graph_integrated_BBB_2014J_day60.ttl"
)


# =========================================================
# Experiment configuration
# =========================================================

CUTOFF_DAY = 60

HIGH_ML_THRESHOLD = 0.70
MODERATE_ML_THRESHOLD = 0.40


# =========================================================
# Namespace
# =========================================================

LS = Namespace(
    "http://example.org/learner-support/"
)


# =========================================================
# Load files
# =========================================================

def load_inputs():
    """
    Load learner profiles, symbolic-rule conclusions and
    trained Logistic Regression model.
    """

    required_files = [
        PROFILE_FILE,
        RULE_RESULTS_FILE,
        MODEL_FILE,
        REASONED_GRAPH_FILE,
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

    rule_results = pd.read_csv(
        RULE_RESULTS_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    return (
        profiles,
        rule_results,
        model,
    )


# =========================================================
# Generate ML probabilities for all learners
# =========================================================

def generate_ml_predictions(
    profiles,
    model,
):
    """
    Generate Day-60 risk probabilities for all 2,292
    learners using the previously trained Logistic
    Regression pipeline.

    Important:
    final_result and risk_target are not passed into the
    model as predictors.

    The held-out test set remains the appropriate dataset
    for reporting predictive performance.
    """

    (
        X,
        y,
        learner_ids,
        model_features,
    ) = prepare_features(
        profiles
    )

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    ml_results = pd.DataFrame(
        {
            "id_student":
                learner_ids.values,

            "ml_risk_probability":
                probabilities,

            "ml_prediction":
                predictions,
        }
    )

    return ml_results


# =========================================================
# Prepare symbolic rule results
# =========================================================

def prepare_rule_results(
    rule_results,
):
    """
    Convert RDF learner identifier:

        Learner_12345

    into:

        id_student = 12345

    so ML and symbolic outputs can be joined.
    """

    rules = rule_results.copy()

    rules[
        "id_student"
    ] = (
        rules[
            "learner"
        ]
        .str.replace(
            "Learner_",
            "",
            regex=False,
        )
        .astype(int)
    )

    return rules


# =========================================================
# ML probability band
# =========================================================

def classify_ml_probability(
    probability,
):
    """
    Create an interpretable probability band.

    >= 0.70     HighMLRisk
    0.40–0.69   ModerateMLRisk
    < 0.40      LowMLRisk
    """

    if probability >= HIGH_ML_THRESHOLD:
        return "HighMLRisk"

    if probability >= MODERATE_ML_THRESHOLD:
        return "ModerateMLRisk"

    return "LowMLRisk"


# =========================================================
# Integrated neuro-symbolic decision
# =========================================================

def integrate_decision(row):
    """
    Combine probabilistic ML evidence with symbolic
    reasoning.

    The ML model does not automatically override symbolic
    evidence.

    The symbolic rules do not automatically override a
    strong probabilistic warning.

    Instead, agreement, disagreement and evidence
    sufficiency are explicitly represented.
    """

    probability = row[
        "ml_risk_probability"
    ]

    symbolic_risk = row[
        "risk_state"
    ]

    symbolic_intervention = row[
        "intervention"
    ]

    evidence = row[
        "evidence_sufficiency"
    ]

    # -----------------------------------------------------
    # I1
    # Strong agreement:
    # ML high + symbolic high
    # -----------------------------------------------------

    if (
        probability >= HIGH_ML_THRESHOLD
        and symbolic_risk == "HighRisk"
    ):

        return pd.Series(
            {
                "final_priority":
                    "HighPriority",

                "final_intervention":
                    "TutorReview",

                "integration_rule":
                    "I1",

                "agreement_state":
                    "MLSymbolicAgreement",

                "integration_explanation":
                    (
                        "The predictive model indicates high "
                        "risk and the symbolic reasoning layer "
                        "also identifies high-risk evidence."
                    ),
            }
        )

    # -----------------------------------------------------
    # I2
    # ML high + symbolic moderate
    #
    # Escalate because the model probability is strong and
    # observable concerns exist.
    # -----------------------------------------------------

    if (
        probability >= HIGH_ML_THRESHOLD
        and symbolic_risk == "ModerateRisk"
    ):

        return pd.Series(
            {
                "final_priority":
                    "HighPriority",

                "final_intervention":
                    "TutorReview",

                "integration_rule":
                    "I2",

                "agreement_state":
                    "PartialAgreement",

                "integration_explanation":
                    (
                        "The predictive model indicates high "
                        "risk while symbolic evidence indicates "
                        "moderate concern. Tutor review is "
                        "recommended."
                    ),
            }
        )

    # -----------------------------------------------------
    # I3
    # High ML probability but symbolic evidence is uncertain.
    #
    # Do not automatically classify the learner as high risk.
    # Escalate to a human because evidence is incomplete.
    # -----------------------------------------------------

    if (
        probability >= HIGH_ML_THRESHOLD
        and symbolic_risk == "UncertainRisk"
    ):

        return pd.Series(
            {
                "final_priority":
                    "HumanReviewPriority",

                "final_intervention":
                    "TutorReview",

                "integration_rule":
                    "I3",

                "agreement_state":
                    "ModelEvidenceUncertainty",

                "integration_explanation":
                    (
                        "The predictive model indicates high "
                        "risk, but the symbolic layer reports "
                        "insufficient or partial evidence. "
                        "Human review is required rather than "
                        "automatic escalation."
                    ),
            }
        )

    # -----------------------------------------------------
    # I4
    # ML high but symbolic system says low risk.
    #
    # This disagreement is itself useful information.
    # -----------------------------------------------------

    if (
        probability >= HIGH_ML_THRESHOLD
        and symbolic_risk == "LowRisk"
    ):

        return pd.Series(
            {
                "final_priority":
                    "HumanReviewPriority",

                "final_intervention":
                    "TutorReview",

                "integration_rule":
                    "I4",

                "agreement_state":
                    "MLSymbolicDisagreement",

                "integration_explanation":
                    (
                        "The predictive model indicates high "
                        "risk but the symbolic evidence appears "
                        "low risk. The disagreement should be "
                        "reviewed by a tutor."
                    ),
            }
        )

    # -----------------------------------------------------
    # I5
    # Symbolic high risk even when ML probability is below
    # the high-risk threshold.
    #
    # Observable evidence remains actionable.
    # -----------------------------------------------------

    if symbolic_risk == "HighRisk":

        return pd.Series(
            {
                "final_priority":
                    "HighPriority",

                "final_intervention":
                    symbolic_intervention,

                "integration_rule":
                    "I5",

                "agreement_state":
                    "SymbolicConcern",

                "integration_explanation":
                    (
                        "The symbolic reasoning layer identifies "
                        "a high-risk evidence pattern even though "
                        "the predictive probability is below the "
                        "high-risk threshold."
                    ),
            }
        )

    # -----------------------------------------------------
    # I6
    # Moderate ML + moderate symbolic concern
    # -----------------------------------------------------

    if (
        probability >= MODERATE_ML_THRESHOLD
        and symbolic_risk == "ModerateRisk"
    ):

        return pd.Series(
            {
                "final_priority":
                    "ModeratePriority",

                "final_intervention":
                    symbolic_intervention,

                "integration_rule":
                    "I6",

                "agreement_state":
                    "MLSymbolicAgreement",

                "integration_explanation":
                    (
                        "Both probabilistic and symbolic "
                        "evidence indicate a moderate level "
                        "of concern."
                    ),
            }
        )

    # -----------------------------------------------------
    # I7
    # Uncertain symbolic evidence
    # -----------------------------------------------------

    if symbolic_risk == "UncertainRisk":

        return pd.Series(
            {
                "final_priority":
                    "MonitoringPriority",

                "final_intervention":
                    "ContinueMonitoring",

                "integration_rule":
                    "I7",

                "agreement_state":
                    "EvidenceUncertainty",

                "integration_explanation":
                    (
                        "The available evidence is insufficient "
                        "or incomplete, so continued monitoring "
                        "is preferred to an automatic risk "
                        "decision."
                    ),
            }
        )

    # -----------------------------------------------------
    # I8
    # ML low + symbolic low
    # -----------------------------------------------------

    if (
        probability < MODERATE_ML_THRESHOLD
        and symbolic_risk == "LowRisk"
    ):

        return pd.Series(
            {
                "final_priority":
                    "LowPriority",

                "final_intervention":
                    "ContinueMonitoring",

                "integration_rule":
                    "I8",

                "agreement_state":
                    "MLSymbolicAgreement",

                "integration_explanation":
                    (
                        "Both the predictive model and symbolic "
                        "reasoning indicate a low level of "
                        "current concern."
                    ),
            }
        )

    # -----------------------------------------------------
    # I9
    # Default moderate case
    # -----------------------------------------------------

    return pd.Series(
        {
            "final_priority":
                "ModeratePriority",

            "final_intervention":
                symbolic_intervention,

            "integration_rule":
                "I9",

            "agreement_state":
                "MixedEvidence",

            "integration_explanation":
                (
                    "The learner presents mixed probabilistic "
                    "and symbolic evidence and should continue "
                    "to be monitored."
                ),
        }
    )


# =========================================================
# Integrate ML + symbolic outputs
# =========================================================

def build_integrated_decisions(
    profiles,
    ml_results,
    rule_results,
):
    """
    Merge learner evidence, ML probability and symbolic
    conclusions into a single decision-support table.
    """

    rules = prepare_rule_results(
        rule_results
    )

    # -----------------------------------------------------
    # Keep selected learner-level evidence
    # -----------------------------------------------------

    evidence_columns = [
        "id_student",
        "weighted_assessment_average",
        "assessment_trend",
        "recent_14_day_clicks",
        "previous_14_day_clicks",
        "engagement_change",
        "days_since_last_activity",
        "assessment_completion_rate",
        "risk_target",
    ]

    evidence = profiles[
        evidence_columns
    ].copy()

    # -----------------------------------------------------
    # Merge numerical evidence + ML
    # -----------------------------------------------------

    integrated = evidence.merge(
        ml_results,
        on="id_student",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------
    # Merge symbolic output
    # -----------------------------------------------------

    rule_columns = [
        "id_student",
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
        "observation",
    ]

    integrated = integrated.merge(
        rules[
            rule_columns
        ],
        on="id_student",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------
    # ML probability band
    # -----------------------------------------------------

    integrated[
        "ml_risk_band"
    ] = integrated[
        "ml_risk_probability"
    ].apply(
        classify_ml_probability
    )

    # -----------------------------------------------------
    # Hybrid decision
    # -----------------------------------------------------

    integration_results = (
        integrated.apply(
            integrate_decision,
            axis=1,
        )
    )

    integrated = pd.concat(
        [
            integrated.reset_index(
                drop=True
            ),
            integration_results.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    return integrated


# =========================================================
# Add ML probability to RDF graph
# =========================================================

def update_rdf_graph(
    integrated_decisions,
):
    """
    Add the ML risk probability to each RDF learner
    observation.

    predictedRiskProbability is already defined in the
    ontology.
    """

    graph = Graph()

    graph.parse(
        REASONED_GRAPH_FILE,
        format="turtle",
    )

    graph.bind(
        "ls",
        LS,
    )

    for _, row in integrated_decisions.iterrows():

        observation = LS[
            row[
                "observation"
            ]
        ]

        graph.set(
            (
                observation,
                LS.predictedRiskProbability,
                Literal(
                    float(
                        row[
                            "ml_risk_probability"
                        ]
                    ),
                    datatype=XSD.decimal,
                ),
            )
        )

    graph.serialize(
        destination=str(
            INTEGRATED_GRAPH_FILE
        ),
        format="turtle",
    )

    return graph


# =========================================================
# Validation
# =========================================================

def validate_integration(
    integrated,
):
    """
    Validate complete hybrid integration.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "NEURO-SYMBOLIC INTEGRATION VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Learners integrated: "
        f"{len(integrated):,}"
    )

    duplicate_count = (
        integrated[
            "id_student"
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate learner rows: "
        f"{duplicate_count}"
    )

    missing_probability = (
        integrated[
            "ml_risk_probability"
        ]
        .isna()
        .sum()
    )

    print(
        f"Missing ML probabilities: "
        f"{missing_probability}"
    )

    missing_symbolic = (
        integrated[
            "risk_state"
        ]
        .isna()
        .sum()
    )

    print(
        f"Missing symbolic decisions: "
        f"{missing_symbolic}"
    )

    print(
        "\nML risk-band distribution"
    )

    print(
        "-" * 50
    )

    print(
        integrated[
            "ml_risk_band"
        ]
        .value_counts()
    )

    print(
        "\nSymbolic risk distribution"
    )

    print(
        "-" * 50
    )

    print(
        integrated[
            "risk_state"
        ]
        .value_counts()
    )

    print(
        "\nFinal priority distribution"
    )

    print(
        "-" * 50
    )

    print(
        integrated[
            "final_priority"
        ]
        .value_counts()
    )

    print(
        "\nFinal intervention distribution"
    )

    print(
        "-" * 50
    )

    print(
        integrated[
            "final_intervention"
        ]
        .value_counts()
    )

    print(
        "\nML / symbolic agreement"
    )

    print(
        "-" * 50
    )

    print(
        integrated[
            "agreement_state"
        ]
        .value_counts()
    )

    print(
        "\nIntegration rule coverage"
    )

    print(
        "-" * 50
    )

    print(
        integrated[
            "integration_rule"
        ]
        .value_counts()
        .sort_index()
    )

    if (
        duplicate_count == 0
        and missing_probability == 0
        and missing_symbolic == 0
    ):

        print(
            "\nIntegration validation: PASS"
        )

    else:

        print(
            "\nIntegration validation: FAIL"
        )


# =========================================================
# Display examples
# =========================================================

def print_examples(
    integrated,
    n=15,
):
    """
    Print representative integrated decisions.
    """

    columns = [
        "id_student",
        "ml_risk_probability",
        "ml_risk_band",
        "performance_state",
        "engagement_state",
        "inactivity_state",
        "risk_state",
        "final_priority",
        "final_intervention",
        "integration_rule",
        "agreement_state",
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXAMPLE INTEGRATED DECISIONS"
    )

    print(
        "=" * 70
    )

    print(
        integrated[
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
# Display disagreements
# =========================================================

def print_disagreement_examples(
    integrated,
    n=10,
):
    """
    Show cases where probabilistic and symbolic reasoning
    disagree.

    These cases are especially relevant to human escalation
    and dissertation analysis.
    """

    disagreements = integrated[
        integrated[
            "agreement_state"
        ]
        == "MLSymbolicDisagreement"
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ML / SYMBOLIC DISAGREEMENT CASES"
    )

    print(
        "=" * 70
    )

    if disagreements.empty:

        print(
            "No direct high-ML / low-symbolic disagreements."
        )

        return

    columns = [
        "id_student",
        "ml_risk_probability",
        "performance_state",
        "engagement_state",
        "inactivity_state",
        "risk_state",
        "final_priority",
        "final_intervention",
    ]

    print(
        disagreements[
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
# Save integrated decisions
# =========================================================

def save_integrated_results(
    integrated,
):
    """
    Save hybrid decision-support output.
    """

    INTEGRATED_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    integrated.to_csv(
        INTEGRATED_OUTPUT_FILE,
        index=False,
    )

    print(
        "\nSaved integrated decisions:"
    )

    print(
        INTEGRATED_OUTPUT_FILE
    )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print(
        "\nRunning neuro-symbolic integration..."
    )

    # -----------------------------------------------------
    # 1. Load inputs
    # -----------------------------------------------------

    (
        profiles,
        rule_results,
        model,
    ) = load_inputs()

    print(
        f"\nLearner profiles loaded: "
        f"{len(profiles):,}"
    )

    print(
        f"Symbolic rule results loaded: "
        f"{len(rule_results):,}"
    )

    # -----------------------------------------------------
    # 2. Generate ML probabilities for every learner
    # -----------------------------------------------------

    ml_results = generate_ml_predictions(
        profiles,
        model,
    )

    print(
        f"ML probabilities generated: "
        f"{len(ml_results):,}"
    )

    # -----------------------------------------------------
    # 3. Integrate ML and symbolic outputs
    # -----------------------------------------------------

    integrated = (
        build_integrated_decisions(
            profiles=profiles,
            ml_results=ml_results,
            rule_results=rule_results,
        )
    )

    # -----------------------------------------------------
    # 4. Validate
    # -----------------------------------------------------

    validate_integration(
        integrated
    )

    # -----------------------------------------------------
    # 5. Display examples
    # -----------------------------------------------------

    print_examples(
        integrated,
        n=15,
    )

    print_disagreement_examples(
        integrated,
        n=10,
    )

    # -----------------------------------------------------
    # 6. Save integrated table
    # -----------------------------------------------------

    save_integrated_results(
        integrated
    )

    # -----------------------------------------------------
    # 7. Add ML probability into RDF graph
    # -----------------------------------------------------

    graph = update_rdf_graph(
        integrated
    )

    print(
        "\nSaved integrated RDF graph:"
    )

    print(
        INTEGRATED_GRAPH_FILE
    )

    print(
        f"\nIntegrated graph triples: "
        f"{len(graph):,}"
    )

    # -----------------------------------------------------
    # 8. Complete
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "NEURO-SYMBOLIC INTEGRATION COMPLETE"
    )

    print(
        "=" * 70
    )