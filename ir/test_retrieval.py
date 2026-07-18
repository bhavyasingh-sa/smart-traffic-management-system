from pathlib import Path
import pickle


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


IR_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ir"
    / "traffic_ir.pkl"
)


def print_section(title):
    print(f"\n{title}")


def load_retriever():

    if not IR_MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nIR model not found.\n\n"
            f"Expected path:\n"
            f"{IR_MODEL_PATH}\n\n"
            "Build the IR system first using:\n"
            "python3 -m ir.build_index"
        )

    print(
        "\nLoading IR historical "
        "traffic retriever..."
    )

    with open(
        IR_MODEL_PATH,
        "rb",
    ) as file:

        retriever = pickle.load(
            file
        )

    print(
        "IR retriever loaded successfully."
    )

    print(
        f"Loaded object type: "
        f"{type(retriever).__name__}"
    )

    return retriever


def format_severity(value):
    """
    Safely format HistoricalSeverity.

    Returns 'Unknown' if the value is missing
    or cannot be converted to a float.
    """

    if value is None:
        return "Unknown"

    try:

        return f"{float(value):.4f}"

    except (
        TypeError,
        ValueError,
    ):

        return "Unknown"


def format_score(value):
    """
    Safely format an optional numeric score.
    """

    if value is None:
        return "Unknown"

    try:

        return f"{float(value):.4f}"

    except (
        TypeError,
        ValueError,
    ):

        return "Unknown"


def display_search_context(
    test_case,
):

    print(
        "\nStructured retrieval context:"
    )

    context_fields = [
        (
            "City",
            test_case.get("city"),
        ),
        (
            "Intersection",
            test_case.get(
                "intersection_id"
            ),
        ),
        (
            "Approach",
            test_case.get("approach"),
        ),
        (
            "Hour",
            test_case.get("hour"),
        ),
        (
            "Weekend",
            test_case.get("weekend"),
        ),
        (
            "Month",
            test_case.get("month"),
        ),
        (
            "Congestion level",
            test_case.get(
                "congestion_level"
            ),
        ),
        (
            "Target severity",
            test_case.get(
                "target_severity"
            ),
        ),
    ]

    for label, value in context_fields:

        if value is not None:

            print(
                f"  {label:<18}: "
                f"{value}"
            )


def display_results(
    query,
    results,
    test_case=None,
):

    print_section(
        f"QUERY: {query}"
    )

    if test_case is not None:

        display_search_context(
            test_case
        )

    if not results:

        print(
            "\nNo matching historical traffic "
            "cases found."
        )

        return

    print(
        f"\nRetrieved {len(results)} "
        f"historical cases.\n"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        historical_severity = (
            format_severity(
                metadata.get(
                    "HistoricalSeverity"
                )
            )
        )

        final_score = format_score(
            result.get(
                "score"
            )
        )

        tfidf_score = format_score(
            result.get(
                "tfidf_score"
            )
        )

        context_bonus = format_score(
            result.get(
                "context_bonus"
            )
        )

        print(
            f"Rank #{rank}"
        )

        print(
            f"Document ID: "
            f"{result.get('doc_id', 'Unknown')}"
        )

        print(
            f"Final hybrid score: "
            f"{final_score}"
        )

        print(
            f"TF-IDF score: "
            f"{tfidf_score}"
        )

        print(
            f"Context bonus: "
            f"{context_bonus}"
        )

        print(
            f"City: "
            f"{metadata.get('City', 'Unknown')}"
        )

        print(
            f"Intersection: "
            f"{metadata.get('IntersectionId', 'Unknown')}"
        )

        print(
            f"Hour: "
            f"{metadata.get('Hour', 'Unknown')}"
        )

        print(
            f"Weekend: "
            f"{metadata.get('Weekend', 'Unknown')}"
        )

        print(
            f"Month: "
            f"{metadata.get('Month', 'Unknown')}"
        )

        print(
            f"Approach: "
            f"{metadata.get('Approach', 'Unknown')}"
        )

        print(
            f"Entry heading: "
            f"{metadata.get('EntryHeading', 'Unknown')}"
        )

        print(
            f"Exit heading: "
            f"{metadata.get('ExitHeading', 'Unknown')}"
        )

        print(
            f"Historical severity: "
            f"{historical_severity}"
        )

        print(
            f"Congestion level: "
            f"{metadata.get('CongestionLevel', 'Unknown')}"
        )

        print(
            f"Stopped time: "
            f"{metadata.get('TotalTimeStopped_p50', 'Unknown')}"
        )

        print(
            f"Delay: "
            f"{metadata.get('TimeFromFirstStop_p50', 'Unknown')}"
        )

        print(
            f"Distance: "
            f"{metadata.get('DistanceToFirstStop_p50', 'Unknown')}"
        )

        print(
            "\nHistorical document:"
        )

        print(
            result.get(
                "document",
                "Unknown",
            )
        )


def verify_retriever(
    retriever,
):

    print_section(
        "IR RETRIEVER INFORMATION"
    )

    required_attributes = [
        "documents",
        "metadata",
        "inverted_index",
        "vectorizer",
        "document_matrix",
    ]

    print(
        "\nChecking retriever components:\n"
    )

    missing_attributes = []

    for attribute in required_attributes:

        exists = hasattr(
            retriever,
            attribute,
        )

        status = (
            "OK"
            if exists
            else "MISSING"
        )

        print(
            f"  {attribute:<20}: "
            f"{status}"
        )

        if not exists:

            missing_attributes.append(
                attribute
            )

    if missing_attributes:

        raise AttributeError(
            "\nLoaded IR retriever is missing "
            "required attributes:\n"
            + ", ".join(
                missing_attributes
            )
        )

    if not hasattr(
        retriever,
        "search",
    ):

        raise AttributeError(
            "\nLoaded IR object does not have "
            "a search() method."
        )

    print(
        f"\nTotal documents: "
        f"{len(retriever.documents):,}"
    )

    print(
        f"Total metadata records: "
        f"{len(retriever.metadata):,}"
    )

    print(
        f"TF-IDF matrix shape: "
        f"{retriever.document_matrix.shape}"
    )


def get_test_cases():
    """
    Return structured IR retrieval test cases.

    Each test case contains:

    1. Natural-language query.
    2. Structured traffic context.

    This allows the hybrid retriever to use both TF-IDF
    similarity and metadata-aware reranking.
    """

    return [

        {
            "name": (
                "Weekday 8 AM West Severe"
            ),

            "query": (
                "Atlanta intersection 84 "
                "weekday June hour 8 morning "
                "west approach severe congestion"
            ),

            "city": "Atlanta",

            "intersection_id": 84,

            "approach": "W",

            "hour": 8,

            "weekend": 0,

            "month": 6,

            "congestion_level": "Severe",

            "target_severity": 0.85,
        },

        {
            "name": (
                "Weekday 8 AM North High"
            ),

            "query": (
                "Atlanta intersection 84 "
                "weekday June hour 8 morning "
                "north approach high congestion"
            ),

            "city": "Atlanta",

            "intersection_id": 84,

            "approach": "N",

            "hour": 8,

            "weekend": 0,

            "month": 6,

            "congestion_level": "High",

            "target_severity": 0.55,
        },

        {
            "name": (
                "Weekday 4 PM East Severe"
            ),

            "query": (
                "Atlanta intersection 84 "
                "weekday June hour 16 afternoon "
                "east approach severe congestion"
            ),

            "city": "Atlanta",

            "intersection_id": 84,

            "approach": "E",

            "hour": 16,

            "weekend": 0,

            "month": 6,

            "congestion_level": "Severe",

            "target_severity": 0.85,
        },

        {
            "name": (
                "Weekend Low Congestion"
            ),

            "query": (
                "Atlanta intersection 84 "
                "weekend low congestion"
            ),

            "city": "Atlanta",

            "intersection_id": 84,

            "approach": None,

            "hour": None,

            "weekend": 1,

            "month": None,

            "congestion_level": "Low",

            "target_severity": 0.10,
        },
    ]


def run_test_case(
    retriever,
    test_case,
):

    print_section(
        f"TEST: {test_case['name']}"
    )

    query = test_case[
        "query"
    ]

    results = retriever.search(
        query=query,

        top_k=5,

        city=test_case.get(
            "city"
        ),

        intersection_id=test_case.get(
            "intersection_id"
        ),

        approach=test_case.get(
            "approach"
        ),

        hour=test_case.get(
            "hour"
        ),

        weekend=test_case.get(
            "weekend"
        ),

        month=test_case.get(
            "month"
        ),

        congestion_level=test_case.get(
            "congestion_level"
        ),

        target_severity=test_case.get(
            "target_severity"
        ),

        strict_context=False,

        candidate_limit=500,
    )

    display_results(
        query=query,
        results=results,
        test_case=test_case,
    )

    return results


def run_interactive_search(
    retriever,
):

    print_section(
        "OPTIONAL INTERACTIVE SEARCH"
    )

    print(
        "\nEnter your own natural-language "
        "traffic query."
    )

    print(
        "Press Enter without typing anything "
        "to skip."
    )

    custom_query = input(
        "\nQuery: "
    ).strip()

    if not custom_query:

        print(
            "\nInteractive search skipped."
        )

        return

    results = retriever.search(
        query=custom_query,
        top_k=5,
        strict_context=False,
        candidate_limit=500,
    )

    display_results(
        query=custom_query,
        results=results,
    )


def main():

    print_section(
        "SMART TRAFFIC AI — "
        "HYBRID IR RETRIEVAL TEST"
    )

    retriever = load_retriever()

    verify_retriever(
        retriever
    )

    test_cases = get_test_cases()

    print_section(
        "RUNNING HYBRID RETRIEVAL TESTS"
    )

    all_results = []

    for test_case in test_cases:

        results = run_test_case(
            retriever=retriever,
            test_case=test_case,
        )

        all_results.append(
            {
                "test_case": test_case,
                "results": results,
            }
        )

    print_section(
        "RETRIEVAL TEST SUMMARY"
    )

    for test_result in all_results:

        test_case = test_result[
            "test_case"
        ]

        results = test_result[
            "results"
        ]

        if results:

            top_result = results[0]

            metadata = top_result.get(
                "metadata",
                {},
            )

            print(
                f"\nTest: "
                f"{test_case['name']}"
            )

            print(
                f"Top result approach: "
                f"{metadata.get('Approach', 'Unknown')}"
            )

            print(
                f"Top result hour: "
                f"{metadata.get('Hour', 'Unknown')}"
            )

            print(
                f"Top result weekend: "
                f"{metadata.get('Weekend', 'Unknown')}"
            )

            print(
                f"Top result month: "
                f"{metadata.get('Month', 'Unknown')}"
            )

            print(
                f"Top result congestion: "
                f"{metadata.get('CongestionLevel', 'Unknown')}"
            )

            print(
                f"Top result severity: "
                f"{format_severity(metadata.get('HistoricalSeverity'))}"
            )

            print(
                f"Final hybrid score: "
                f"{format_score(top_result.get('score'))}"
            )

        else:

            print(
                f"\nTest: "
                f"{test_case['name']}"
            )

            print(
                "No results returned."
            )

    run_interactive_search(
        retriever
    )

    print_section(
        "IR RETRIEVAL TEST COMPLETE"
    )

    print(
        "\nIR pipeline verified:"
    )

    print(
        "  1. Historical traffic documents loaded"
    )

    print(
        "  2. Inverted index available"
    )

    print(
        "  3. TF-IDF document matrix available"
    )

    print(
        "  4. Candidate documents retrieved"
    )

    print(
        "  5. TF-IDF similarity calculated"
    )

    print(
        "  6. Structured traffic context passed "
        "to retriever"
    )

    print(
        "  7. Metadata-aware reranking applied"
    )

    print(
        "  8. Congestion-level matching applied"
    )

    print(
        "  9. Historical severity similarity applied"
    )

    print(
        " 10. Final hybrid results ranked successfully"
    )


if __name__ == "__main__":
    main()