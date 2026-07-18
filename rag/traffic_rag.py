from simulation.movement_definitions import (
    TRAVEL_DIRECTIONS as APPROACHES,
)

from simulation.controller_core import (
    CITY,
    INTERSECTION_ID,
    SIMULATION_HOUR,
    WEEKEND,
    MONTH,
    load_ml_model,
    load_ir_retriever,
    build_ml_predictions,
    build_ir_predictions,
)

from rag.generator import (
    generate_traffic_explanation,
)


def print_section(title):
    print(f"\n{title}")


def generate_approach_analysis(
    approach,
    ml_prediction,
    ir_prediction,
):
    """
    Generate a Gemini RAG explanation for one traffic approach.

    The explanation is grounded using:
    1. ML congestion class
    2. ML severity score
    3. IR historical severity score
    4. Retrieved historical traffic cases
    """

    ml_class = ml_prediction[
        "class"
    ]

    ml_severity = float(
        ml_prediction[
            "severity"
        ]
    )

    ir_severity = float(
        ir_prediction[
            "severity"
        ]
    )

    retrieved_cases = ir_prediction.get(
        "top_results",
        [],
    )

    explanation = (
        generate_traffic_explanation(
            approach=approach,
            ml_class=ml_class,
            ml_severity=ml_severity,
            ir_severity=ir_severity,
            retrieved_cases=retrieved_cases,
        )
    )

    return explanation


def run_traffic_rag(
    hour=SIMULATION_HOUR,
    weekend=WEEKEND,
    month=MONTH,
):
    """
    Run the complete Smart Traffic AI RAG pipeline.

    Pipeline:
    1. Load trained ML model.
    2. Load IR historical traffic retriever.
    3. Generate ML predictions.
    4. Retrieve relevant historical traffic cases.
    5. Generate grounded Gemini explanations.
    """

    print_section(
        "SMART TRAFFIC AI — GEMINI RAG ANALYSIS"
    )

    print(
        f"\nCity: {CITY}"
    )

    print(
        f"Intersection: {INTERSECTION_ID}"
    )

    print(
        f"Hour: {hour}:00"
    )

    print(
        f"Weekend: {weekend}"
    )

    print(
        f"Month: {month}"
    )

    model_bundle = load_ml_model()

    retriever = load_ir_retriever()

    print_section(
        "GENERATING ML PREDICTIONS"
    )

    ml_predictions = (
        build_ml_predictions(
            model_bundle=model_bundle,
            hour=hour,
            weekend=weekend,
            month=month,
        )
    )

    print_section(
        "RETRIEVING HISTORICAL TRAFFIC CASES"
    )

    ir_predictions = (
        build_ir_predictions(
            retriever=retriever,
            hour=hour,
            weekend=weekend,
            month=month,
            ml_predictions=ml_predictions,
            top_k=5,
        )
    )

    print_section(
        "ML + IR TRAFFIC EVIDENCE"
    )

    print(
        "\nApproach | ML Class   | "
        "ML Severity | IR Severity | IR Cases"
    )

    for approach in APPROACHES:

        ml_data = ml_predictions[
            approach
        ]

        ir_data = ir_predictions[
            approach
        ]

        print(
            f"   {approach}     | "
            f"{ml_data['class']:<10} | "
            f"{ml_data['severity']:>11.4f} | "
            f"{ir_data['severity']:>11.4f} | "
            f"{ir_data['retrieved_cases']:>8}"
        )

    print_section(
        "GENERATING GROUNDED GEMINI EXPLANATIONS"
    )

    explanations = {}

    for approach in APPROACHES:

        print(
            f"\nGenerating explanation for "
            f"approach {approach}..."
        )

        try:

            explanation = (
                generate_approach_analysis(
                    approach=approach,
                    ml_prediction=(
                        ml_predictions[
                            approach
                        ]
                    ),
                    ir_prediction=(
                        ir_predictions[
                            approach
                        ]
                    ),
                )
            )

        except Exception as error:

            explanation = (
                "Gemini explanation generation "
                f"failed: {error}"
            )

        explanations[
            approach
        ] = explanation

    print_section(
        "FINAL GROUNDED TRAFFIC ANALYSIS"
    )

    for approach in APPROACHES:

        ml_data = ml_predictions[
            approach
        ]

        ir_data = ir_predictions[
            approach
        ]

        print(
            f"\nAPPROACH: {approach}"
        )

        print(
            f"ML class: "
            f"{ml_data['class']}"
        )

        print(
            f"ML severity: "
            f"{ml_data['severity']:.4f}"
        )

        print(
            f"IR severity: "
            f"{ir_data['severity']:.4f}"
        )

        print(
            f"Historical cases used: "
            f"{ir_data['retrieved_cases']}"
        )

        print(
            "\nGemini RAG explanation:\n"
        )

        print(
            explanations[
                approach
            ]
        )

    combined_scores = {}

    for approach in APPROACHES:

        ml_severity = float(
            ml_predictions[
                approach
            ]["severity"]
        )

        ir_severity = float(
            ir_predictions[
                approach
            ]["severity"]
        )

        combined_scores[
            approach
        ] = (
            0.60 * ml_severity
            + 0.40 * ir_severity
        )

    highest_priority_approach = max(
        combined_scores,
        key=combined_scores.get,
    )

    print_section(
        "RAG PRIORITY SUMMARY"
    )

    print(
        "\nCombined evidence scores:"
    )

    for approach in APPROACHES:

        print(
            f"  {approach}: "
            f"{combined_scores[approach]:.4f}"
        )

    print(
        f"\nHighest evidence-based priority: "
        f"{highest_priority_approach}"
    )

    print(
        "\nNote: This RAG evidence score is used "
        "for explanation and analysis. The actual "
        "adaptive controller still uses its full "
        "priority function with live queue, waiting "
        "time, ML severity, IR evidence, and "
        "starvation prevention."
    )

    return {
        "city": CITY,
        "intersection_id": INTERSECTION_ID,
        "hour": hour,
        "weekend": weekend,
        "month": month,
        "ml_predictions": ml_predictions,
        "ir_predictions": ir_predictions,
        "explanations": explanations,
        "combined_scores": combined_scores,
        "highest_priority_approach": (
            highest_priority_approach
        ),
    }


def main():

    run_traffic_rag(
        hour=SIMULATION_HOUR,
        weekend=WEEKEND,
        month=MONTH,
    )


if __name__ == "__main__":
    main()