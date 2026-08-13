from pathlib import Path

import pandas as pd
from rdflib import Graph, Namespace, RDF


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


# ---------------------------------------------------------
# Namespace
# ---------------------------------------------------------

LS = Namespace(
    "http://example.org/learner-support/"
)


# ---------------------------------------------------------
# Load graph
# ---------------------------------------------------------

def load_graph():
    """
    Load the RDF learner knowledge graph.
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
# Helper
# ---------------------------------------------------------

def local_name(uri):
    """
    Extract local RDF identifier.

    Example:
        http://example.org/learner-support/Inactive
        -> Inactive
    """

    return str(uri).split("/")[-1]


# ---------------------------------------------------------
# Extract observations from graph
# ---------------------------------------------------------

def extract_observations(graph):
    """
    Read the symbolic evidence associated with each learner
    observation from the RDF graph.
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

    results = []

    for row in graph.query(query):

        results.append(
            {
                "learner":
                    local_name(row.learner),

                "observation":
                    local_name(row.observation),

                "performance_state":
                    local_name(row.performance),

                "assessment_trend_state":
                    local_name(row.trend),

                "engagement_state":
                    local_name(row.engagement),

                "inactivity_state":
                    local_name(row.inactivity),

                "completion_state":
                    local_name(row.completion),

                "submission_state":
                    local_name(row.submission),

                "evidence_sufficiency":
                    local_name(row.evidence),
            }
        )

    return pd.DataFrame(
        results
    )


# ---------------------------------------------------------
# Symbolic rule engine
# ---------------------------------------------------------

def apply_symbolic_rules(row):
    """
    Apply transparent intervention rules.

    Rule ordering matters:
    the first matching rule is selected.

    Output:
        risk_state
        intervention
        rule_id
        rule_explanation
    """

    performance = row[
        "performance_state"
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

    trend = row[
        "assessment_trend_state"
    ]

    # -----------------------------------------------------
    # Rule 1
    #
    # Insufficient evidence should not automatically be
    # interpreted as high risk.
    # -----------------------------------------------------

    if evidence == "InsufficientEvidence":

        return {
            "risk_state":
                "UncertainRisk",

            "intervention":
                "ContinueMonitoring",

            "rule_id":
                "R1",

            "rule_explanation":
                (
                    "Insufficient assessment and engagement "
                    "evidence is available for a reliable "
                    "intervention decision."
                ),
        }

    # -----------------------------------------------------
    # Rule 2
    #
    # Partial evidence means the system should remain
    # cautious.
    # -----------------------------------------------------

    if evidence == "PartialEvidence":

        return {
            "risk_state":
                "UncertainRisk",

            "intervention":
                "ContinueMonitoring",

            "rule_id":
                "R2",

            "rule_explanation":
                (
                    "Only partial learner evidence is "
                    "available, so continued monitoring is "
                    "recommended before stronger action."
                ),
        }

    # -----------------------------------------------------
    # Rule 3
    #
    # No assessment submission + high inactivity
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
                "R3",

            "rule_explanation":
                (
                    "The learner has submitted no required "
                    "assessment and is highly inactive."
                ),
        }

    # -----------------------------------------------------
    # Rule 4
    #
    # Low/borderline performance + sharply declining
    # engagement + inactivity
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
                "R4",

            "rule_explanation":
                (
                    "Low or borderline performance is "
                    "combined with sharply declining "
                    "engagement and sustained inactivity."
                ),
        }

    # -----------------------------------------------------
    # Rule 5
    #
    # Strongly declining assessment + strongly declining
    # engagement
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
                "R5",

            "rule_explanation":
                (
                    "Assessment performance and engagement "
                    "are both declining sharply."
                ),
        }

    # -----------------------------------------------------
    # Rule 6
    #
    # Satisfactory performance but engagement decline.
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
                "R6",

            "rule_explanation":
                (
                    "Current performance is satisfactory, "
                    "but engagement is declining."
                ),
        }

    # -----------------------------------------------------
    # Rule 7
    #
    # Inactivity concern
    # -----------------------------------------------------

    if inactivity == "HighlyInactive":

        return {
            "risk_state":
                "ModerateRisk",

            "intervention":
                "AutomatedReminder",

            "rule_id":
                "R7",

            "rule_explanation":
                (
                    "The learner has been inactive for an "
                    "extended period."
                ),
        }

    # -----------------------------------------------------
    # Rule 8
    #
    # Healthy profile
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
                "R8",

            "rule_explanation":
                (
                    "The learner shows satisfactory or "
                    "strong performance, is active, and has "
                    "completed the required assessments."
                ),
        }

    # -----------------------------------------------------
    # Default rule
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
                "not meet a higher-priority intervention "
                "rule."
            ),
    }


# ---------------------------------------------------------
# Apply rules to all learners
# ---------------------------------------------------------

def run_rule_engine(observations):
    """
    Apply the symbolic rules to every learner observation.
    """

    outputs = observations.apply(
        apply_symbolic_rules,
        axis=1,
    )

    outputs = pd.DataFrame(
        outputs.tolist()
    )

    return pd.concat(
        [
            observations.reset_index(
                drop=True
            ),
            outputs.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )


# ---------------------------------------------------------
# Add rule conclusions back to RDF graph
# ---------------------------------------------------------

def add_rule_results_to_graph(
    graph,
    rule_results,
):
    """
    Add symbolic risk-state conclusions to the RDF graph.

    Example:

        Observation_123_D60
            hasRiskState
            HighRisk
    """

    for _, row in rule_results.iterrows():

        observation = LS[
            row["observation"]
        ]

        risk_state = LS[
            row["risk_state"]
        ]

        intervention = LS[
            row["intervention"]
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
# Validation
# ---------------------------------------------------------

def validate_rule_results(
    rule_results,
):
    """
    Display rule coverage and risk-state distributions.
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

    print(
        "\nRule coverage"
    )

    print(
        "-" * 50
    )

    print(
        rule_results[
            "rule_id"
        ]
        .value_counts()
        .sort_index()
    )


# ---------------------------------------------------------
# Show example rule decisions
# ---------------------------------------------------------

def print_examples(
    rule_results,
    n=15,
):
    """
    Print learner-level symbolic reasoning examples.
    """

    columns = [
        "learner",
        "performance_state",
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
        .head(n)
        .to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# Save updated graph
# ---------------------------------------------------------

def save_reasoned_graph(
    graph,
):
    """
    Save graph after adding symbolic reasoning results.
    """

    output_file = (
        PROJECT_ROOT
        / "outputs"
        / "graphs"
        / "learner_graph_reasoned_BBB_2014J_day60.ttl"
    )

    graph.serialize(
        destination=str(
            output_file
        ),
        format="turtle",
    )

    print(
        "\nSaved reasoned knowledge graph:"
    )

    print(
        output_file
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "\nRunning symbolic intervention rules..."
    )

    # -----------------------------------------------------
    # 1. Load knowledge graph
    # -----------------------------------------------------

    graph = load_graph()

    print(
        f"Knowledge graph triples loaded: "
        f"{len(graph):,}"
    )

    # -----------------------------------------------------
    # 2. Extract learner observations
    # -----------------------------------------------------

    observations = extract_observations(
        graph
    )

    print(
        f"Learner observations extracted: "
        f"{len(observations):,}"
    )

    # -----------------------------------------------------
    # 3. Run symbolic rules
    # -----------------------------------------------------

    rule_results = run_rule_engine(
        observations
    )

    # -----------------------------------------------------
    # 4. Validate
    # -----------------------------------------------------

    validate_rule_results(
        rule_results
    )

    print_examples(
        rule_results
    )

    # -----------------------------------------------------
    # 5. Save tabular rule results
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # 6. Add conclusions to RDF graph
    # -----------------------------------------------------

    graph = add_rule_results_to_graph(
        graph,
        rule_results,
    )

    # -----------------------------------------------------
    # 7. Save updated knowledge graph
    # -----------------------------------------------------

    save_reasoned_graph(
        graph
    )

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