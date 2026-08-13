from pathlib import Path

import pandas as pd
from rdflib import Graph, Namespace


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GRAPH_INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "graphs"
    / "learner_graph_BBB_2014J_day60.ttl"
)

RULE_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "learner_rule_results_BBB_2014J_day60.csv"
)

REASONED_GRAPH_OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "graphs"
    / "learner_graph_reasoned_BBB_2014J_day60.ttl"
)


# ---------------------------------------------------------
# Namespace
# ---------------------------------------------------------

LS = Namespace(
    "http://example.org/learner-support/"
)


# ---------------------------------------------------------
# Load RDF knowledge graph
# ---------------------------------------------------------

def load_graph():
    """
    Load the RDF learner knowledge graph produced by
    src.knowledge_graph.
    """

    if not GRAPH_INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Knowledge graph not found:\n"
            f"{GRAPH_INPUT_FILE}\n\n"
            f"Run:\n"
            f"python -m src.knowledge_graph"
        )

    graph = Graph()

    graph.parse(
        GRAPH_INPUT_FILE,
        format="turtle",
    )

    graph.bind(
        "ls",
        LS,
    )

    return graph


# ---------------------------------------------------------
# URI helper
# ---------------------------------------------------------

def local_name(uri):
    """
    Extract the final RDF identifier from a URI.

    Example:

        http://example.org/learner-support/Inactive

    becomes:

        Inactive
    """

    return str(uri).split("/")[-1]


# ---------------------------------------------------------
# Extract learner evidence from RDF graph
# ---------------------------------------------------------

def extract_observations(graph):
    """
    Query the knowledge graph and reconstruct one symbolic
    evidence row per learner observation.
    """

    query = """
    PREFIX ls: <http://example.org/learner-support/>

    SELECT
        ?learner
        ?observation
        ?performance
        ?trend
        ?engagement
        ?inactivity
        ?completion
        ?submission
        ?evidence

    WHERE {

        ?learner
            a ls:Learner ;
            ls:hasObservation ?observation .

        ?observation
            ls:hasPerformanceState ?performance ;
            ls:hasAssessmentTrendState ?trend ;
            ls:hasEngagementState ?engagement ;
            ls:hasInactivityState ?inactivity ;
            ls:hasCompletionState ?completion ;
            ls:hasSubmissionState ?submission ;
            ls:hasEvidenceSufficiency ?evidence .
    }
    """

    rows = []

    for row in graph.query(query):

        rows.append(
            {
                "learner":
                    local_name(
                        row.learner
                    ),

                "observation":
                    local_name(
                        row.observation
                    ),

                "performance_state":
                    local_name(
                        row.performance
                    ),

                "assessment_trend_state":
                    local_name(
                        row.trend
                    ),

                "engagement_state":
                    local_name(
                        row.engagement
                    ),

                "inactivity_state":
                    local_name(
                        row.inactivity
                    ),

                "completion_state":
                    local_name(
                        row.completion
                    ),

                "submission_state":
                    local_name(
                        row.submission
                    ),

                "evidence_sufficiency":
                    local_name(
                        row.evidence
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------
# Symbolic rule engine
# ---------------------------------------------------------

def apply_symbolic_rules(row):
    """
    Apply transparent symbolic learner-support rules.

    Rule priority:

        1. Specific high-risk patterns
        2. Moderate-risk observable patterns
        3. Healthy / low-risk patterns
        4. Evidence-sufficiency fallbacks
        5. Default fallback

    The first matching rule is returned.

    Output:
        risk_state
        intervention
        rule_id
        rule_explanation
    """

    performance = row[
        "performance_state"
    ]

    trend = row[
        "assessment_trend_state"
    ]

    engagement = row[
        "engagement_state"
    ]

    inactivity = row[
        "inactivity_state"
    ]

    completion = row[
        "completion_state"
    ]

    evidence = row[
        "evidence_sufficiency"
    ]

    # -----------------------------------------------------
    # R1
    #
    # No assessment submission + prolonged inactivity
    #
    # This is evaluated before evidence-sufficiency rules.
    # -----------------------------------------------------

    if (
        completion == "NoAssessmentSubmission"
        and inactivity == "HighlyInactive"
    ):

        return {
            "risk_state":
                "HighRisk",

            "intervention":
                "TutorReview",

            "rule_id":
                "R1",

            "rule_explanation":
                (
                    "The learner has submitted no required "
                    "assessment and has been highly inactive."
                ),
        }

    # -----------------------------------------------------
    # R2
    #
    # Weak performance + sharply declining engagement
    # + sustained inactivity
    # -----------------------------------------------------

    if (
        performance
        in [
            "LowPerformance",
            "BorderlinePerformance",
        ]
        and engagement
        == "SharplyDecliningEngagement"
        and inactivity
        in [
            "Inactive",
            "HighlyInactive",
        ]
    ):

        return {
            "risk_state":
                "HighRisk",

            "intervention":
                "TutorReview",

            "rule_id":
                "R2",

            "rule_explanation":
                (
                    "Low or borderline performance is "
                    "combined with sharply declining "
                    "engagement and sustained inactivity."
                ),
        }

    # -----------------------------------------------------
    # R3
    #
    # Assessment and engagement are both sharply declining.
    # -----------------------------------------------------

    if (
        trend
        == "SharplyDecliningAssessment"
        and engagement
        == "SharplyDecliningEngagement"
    ):

        return {
            "risk_state":
                "HighRisk",

            "intervention":
                "ContactLearner",

            "rule_id":
                "R3",

            "rule_explanation":
                (
                    "Assessment performance and engagement "
                    "are both declining sharply."
                ),
        }

    # -----------------------------------------------------
    # R4
    #
    # Performance currently acceptable, but engagement is
    # deteriorating.
    # -----------------------------------------------------

    if (
        performance
        == "SatisfactoryPerformance"
        and engagement
        in [
            "DecliningEngagement",
            "SharplyDecliningEngagement",
        ]
    ):

        return {
            "risk_state":
                "ModerateRisk",

            "intervention":
                "ContactLearner",

            "rule_id":
                "R4",

            "rule_explanation":
                (
                    "Current performance is satisfactory, "
                    "but learner engagement is declining."
                ),
        }

    # -----------------------------------------------------
    # R5
    #
    # Highly inactive learner.
    # -----------------------------------------------------

    if (
        inactivity
        == "HighlyInactive"
    ):

        return {
            "risk_state":
                "ModerateRisk",

            "intervention":
                "AutomatedReminder",

            "rule_id":
                "R5",

            "rule_explanation":
                (
                    "The learner has been inactive for an "
                    "extended period."
                ),
        }

    # -----------------------------------------------------
    # R6
    #
    # Healthy profile.
    # -----------------------------------------------------

    if (
        performance
        in [
            "StrongPerformance",
            "SatisfactoryPerformance",
        ]
        and inactivity == "Active"
        and completion == "Complete"
        and engagement
        not in [
            "SharplyDecliningEngagement",
        ]
    ):

        return {
            "risk_state":
                "LowRisk",

            "intervention":
                "ContinueMonitoring",

            "rule_id":
                "R6",

            "rule_explanation":
                (
                    "The learner shows satisfactory or "
                    "strong performance, remains active, and "
                    "has completed the required assessments."
                ),
        }

    # -----------------------------------------------------
    # R7
    #
    # Insufficient evidence fallback.
    #
    # Missing data must not automatically become high risk.
    # -----------------------------------------------------

    if (
        evidence
        == "InsufficientEvidence"
    ):

        return {
            "risk_state":
                "UncertainRisk",

            "intervention":
                "ContinueMonitoring",

            "rule_id":
                "R7",

            "rule_explanation":
                (
                    "Insufficient assessment and engagement "
                    "evidence is available for a reliable "
                    "intervention decision."
                ),
        }

    # -----------------------------------------------------
    # R8
    #
    # Partial evidence fallback.
    # -----------------------------------------------------

    if (
        evidence
        == "PartialEvidence"
    ):

        return {
            "risk_state":
                "UncertainRisk",

            "intervention":
                "ContinueMonitoring",

            "rule_id":
                "R8",

            "rule_explanation":
                (
                    "Only partial learner evidence is "
                    "available, so continued monitoring is "
                    "recommended before stronger action."
                ),
        }

    # -----------------------------------------------------
    # R9
    #
    # Default fallback.
    # -----------------------------------------------------

    return {
        "risk_state":
            "ModerateRisk",

        "intervention":
            "ContinueMonitoring",

        "rule_id":
            "R9",

        "rule_explanation":
            (
                "The learner shows some indicators that do "
                "not meet the criteria for a higher-priority "
                "or low-risk rule."
            ),
    }


# ---------------------------------------------------------
# Run rule engine
# ---------------------------------------------------------

def run_rule_engine(
    observations,
):
    """
    Apply the symbolic rules to every learner observation.
    """

    rule_outputs = observations.apply(
        apply_symbolic_rules,
        axis=1,
    )

    rule_outputs = pd.DataFrame(
        rule_outputs.tolist()
    )

    combined = pd.concat(
        [
            observations.reset_index(
                drop=True
            ),
            rule_outputs.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    return combined


# ---------------------------------------------------------
# Add conclusions to RDF graph
# ---------------------------------------------------------

def add_rule_results_to_graph(
    graph,
    rule_results,
):
    """
    Add symbolic risk-state and intervention conclusions
    back into the RDF knowledge graph.

    Example:

        Observation_123_D60
            hasRiskState
            HighRisk
    """

    for _, row in rule_results.iterrows():

        observation = LS[
            row[
                "observation"
            ]
        ]

        risk_state = LS[
            row[
                "risk_state"
            ]
        ]

        intervention = LS[
            row[
                "intervention"
            ]
        ]

        graph.add(
            (
                observation,
                LS.hasRiskState,
                risk_state,
            )
        )

        graph.add(
            (
                risk_state,
                LS.supportsIntervention,
                intervention,
            )
        )

    return graph


# ---------------------------------------------------------
# Validate rule results
# ---------------------------------------------------------

def validate_rule_results(
    rule_results,
):
    """
    Display rule coverage, risk-state distribution and
    intervention distribution.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SYMBOLIC RULE ENGINE VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Learners processed: "
        f"{len(rule_results):,}"
    )

    # -----------------------------------------------------
    # Risk distribution
    # -----------------------------------------------------

    print(
        "\nRisk-state distribution"
    )

    print(
        "-" * 50
    )

    print(
        rule_results[
            "risk_state"
        ]
        .value_counts()
    )

    # -----------------------------------------------------
    # Intervention distribution
    # -----------------------------------------------------

    print(
        "\nIntervention distribution"
    )

    print(
        "-" * 50
    )

    print(
        rule_results[
            "intervention"
        ]
        .value_counts()
    )

    # -----------------------------------------------------
    # Rule coverage
    # -----------------------------------------------------

    print(
        "\nRule coverage"
    )

    print(
        "-" * 50
    )

    coverage = (
        rule_results[
            "rule_id"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        coverage
    )

    # -----------------------------------------------------
    # Check whether every learner received exactly one rule
    # -----------------------------------------------------

    if len(rule_results) > 0:

        missing_rules = (
            rule_results[
                "rule_id"
            ]
            .isna()
            .sum()
        )

        print(
            f"\nLearners without a rule: "
            f"{missing_rules}"
        )

        if missing_rules == 0:

            print(
                "Rule assignment check: PASS"
            )

        else:

            print(
                "Rule assignment check: FAIL"
            )


# ---------------------------------------------------------
# Print examples
# ---------------------------------------------------------

def print_examples(
    rule_results,
    n=15,
):
    """
    Print a sample of learner decisions.
    """

    columns = [
        "learner",
        "performance_state",
        "assessment_trend_state",
        "engagement_state",
        "inactivity_state",
        "completion_state",
        "evidence_sufficiency",
        "risk_state",
        "intervention",
        "rule_id",
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXAMPLE SYMBOLIC DECISIONS"
    )

    print(
        "=" * 70
    )

    print(
        rule_results[
            columns
        ]
        .head(
            n
        )
        .to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# Print examples specifically for TutorReview
# ---------------------------------------------------------

def print_tutor_review_examples(
    rule_results,
    n=10,
):
    """
    Print several high-priority tutor-review cases.
    """

    tutor_cases = rule_results[
        rule_results[
            "intervention"
        ]
        == "TutorReview"
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXAMPLE TUTOR REVIEW CASES"
    )

    print(
        "=" * 70
    )

    if tutor_cases.empty:

        print(
            "No TutorReview cases were generated."
        )

        return

    columns = [
        "learner",
        "performance_state",
        "engagement_state",
        "inactivity_state",
        "completion_state",
        "evidence_sufficiency",
        "risk_state",
        "rule_id",
        "rule_explanation",
    ]

    print(
        tutor_cases[
            columns
        ]
        .head(
            n
        )
        .to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# Save tabular rule results
# ---------------------------------------------------------

def save_rule_results(
    rule_results,
):
    """
    Save learner-level symbolic reasoning results.
    """

    RULE_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rule_results.to_csv(
        RULE_OUTPUT_FILE,
        index=False,
    )

    print(
        "\nSaved rule results:"
    )

    print(
        RULE_OUTPUT_FILE
    )


# ---------------------------------------------------------
# Save updated RDF graph
# ---------------------------------------------------------

def save_reasoned_graph(
    graph,
):
    """
    Serialize the knowledge graph after symbolic rule
    conclusions have been added.
    """

    REASONED_GRAPH_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph.serialize(
        destination=str(
            REASONED_GRAPH_OUTPUT_FILE
        ),
        format="turtle",
    )

    print(
        "\nSaved reasoned knowledge graph:"
    )

    print(
        REASONED_GRAPH_OUTPUT_FILE
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "\nRunning symbolic intervention rules..."
    )

    # -----------------------------------------------------
    # 1. Load RDF knowledge graph
    # -----------------------------------------------------

    graph = load_graph()

    print(
        f"Knowledge graph triples loaded: "
        f"{len(graph):,}"
    )

    # -----------------------------------------------------
    # 2. Extract learner evidence
    # -----------------------------------------------------

    observations = extract_observations(
        graph
    )

    print(
        f"Learner observations extracted: "
        f"{len(observations):,}"
    )

    # -----------------------------------------------------
    # 3. Run rule engine
    # -----------------------------------------------------

    rule_results = run_rule_engine(
        observations
    )

    # -----------------------------------------------------
    # 4. Validate rule behaviour
    # -----------------------------------------------------

    validate_rule_results(
        rule_results
    )

    # -----------------------------------------------------
    # 5. Display examples
    # -----------------------------------------------------

    print_examples(
        rule_results,
        n=15,
    )

    print_tutor_review_examples(
        rule_results,
        n=10,
    )

    # -----------------------------------------------------
    # 6. Save tabular results
    # -----------------------------------------------------

    save_rule_results(
        rule_results
    )

    # -----------------------------------------------------
    # 7. Add rule conclusions to RDF graph
    # -----------------------------------------------------

    graph = add_rule_results_to_graph(
        graph,
        rule_results,
    )

    # -----------------------------------------------------
    # 8. Save reasoned graph
    # -----------------------------------------------------

    save_reasoned_graph(
        graph
    )

    # -----------------------------------------------------
    # 9. Complete
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SYMBOLIC REASONING COMPLETE"
    )

    print(
        "=" * 70
    )