from pathlib import Path
import pickle

import pandas as pd

from ir.document_builder import build_documents
from ir.inverted_index import InvertedIndex
from ir.retriever import TrafficRetriever

from ml.features import (
    calculate_scale_limits,
    create_congestion_score,
    congestion_level,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train.csv"
)

INDEX_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "ir"
)

INDEX_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

IR_MODEL_PATH = (
    INDEX_DIRECTORY
    / "traffic_ir.pkl"
)


MAX_GENERAL_RECORDS = 40_000

SELECTED_CITY = "Atlanta"

SELECTED_INTERSECTION_ID = 84

RANDOM_STATE = 42


# Must match simulation/build_profiles.py exactly.
APPROACH_MAPPING = {
    "N": "N",
    "NE": "N",

    "S": "S",
    "SW": "S",

    "E": "E",
    "SE": "E",

    "W": "W",
    "NW": "W",
}


def print_section(title):
    print(f"\n{title}")


print_section(
    "SMART TRAFFIC AI — IR INDEX BUILDER"
)


print("\nLoading train.csv...")

train = pd.read_csv(
    TRAIN_PATH
)

print(
    f"Loaded {len(train):,} "
    f"historical records."
)


selected_intersection = train[
    (
        train["City"]
        == SELECTED_CITY
    )
    &
    (
        train["IntersectionId"]
        == SELECTED_INTERSECTION_ID
    )
].copy()


print(
    f"\nSelected intersection records: "
    f"{len(selected_intersection):,}"
)


if selected_intersection.empty:

    raise ValueError(
        "Selected intersection was not found."
    )


print_section(
    "CALCULATING LOCAL HISTORICAL SEVERITY"
)


scale_limits = calculate_scale_limits(
    selected_intersection
)


print(
    "\nLocal scale limits:"
)


for key, value in scale_limits.items():

    print(
        f"  {key}: {value:.4f}"
    )


selected_intersection[
    "HistoricalSeverity"
] = create_congestion_score(
    selected_intersection,
    scale_limits,
)


selected_intersection[
    "CongestionLevel"
] = selected_intersection[
    "HistoricalSeverity"
].apply(
    congestion_level
)


selected_intersection[
    "Approach"
] = (
    selected_intersection[
        "EntryHeading"
    ]
    .astype(str)
    .str.upper()
    .str.strip()
    .map(APPROACH_MAPPING)
)


selected_intersection = (
    selected_intersection[
        selected_intersection[
            "Approach"
        ].notna()
    ]
    .copy()
)


print(
    f"\nSelected intersection records "
    f"after approach mapping: "
    f"{len(selected_intersection):,}"
)


print(
    "\nApproach distribution:"
)


for approach in [
    "N",
    "S",
    "E",
    "W",
]:

    count = int(
        (
            selected_intersection[
                "Approach"
            ]
            == approach
        ).sum()
    )

    print(
        f"  {approach}: "
        f"{count:,} records"
    )


other_records = train.drop(
    selected_intersection.index,
    errors="ignore",
).copy()


sample_size = min(
    MAX_GENERAL_RECORDS,
    len(other_records),
)


general_sample = other_records.sample(
    n=sample_size,
    random_state=RANDOM_STATE,
).copy()


general_scale_limits = (
    calculate_scale_limits(
        general_sample
    )
)


general_sample[
    "HistoricalSeverity"
] = create_congestion_score(
    general_sample,
    general_scale_limits,
)


general_sample[
    "CongestionLevel"
] = general_sample[
    "HistoricalSeverity"
].apply(
    congestion_level
)


general_sample[
    "Approach"
] = (
    general_sample[
        "EntryHeading"
    ]
    .astype(str)
    .str.upper()
    .str.strip()
    .map(APPROACH_MAPPING)
)


general_sample = (
    general_sample[
        general_sample[
            "Approach"
        ].notna()
    ]
    .copy()
)


knowledge_base = pd.concat(
    [
        selected_intersection,
        general_sample,
    ],
    ignore_index=True,
)


knowledge_base = (
    knowledge_base
    .drop_duplicates(
        subset=["RowId"]
    )
    .reset_index(drop=True)
)


print_section(
    "KNOWLEDGE BASE"
)


print(
    f"\nKnowledge base size: "
    f"{len(knowledge_base):,}"
)


selected_count = int(
    (
        (
            knowledge_base["City"]
            == SELECTED_CITY
        )
        &
        (
            knowledge_base[
                "IntersectionId"
            ]
            == SELECTED_INTERSECTION_ID
        )
    ).sum()
)


print(
    f"Selected intersection records "
    f"in knowledge base: "
    f"{selected_count:,}"
)


print_section(
    "BUILDING SEARCHABLE DOCUMENTS"
)


print(
    "\nConverting structured traffic "
    "records into searchable documents..."
)


documents = build_documents(
    knowledge_base
)


print(
    f"Created {len(documents):,} "
    f"traffic documents."
)


print(
    "\nExample document:"
)

print(
    documents[0]
)


metadata_columns = [
    "RowId",
    "IntersectionId",
    "City",
    "Latitude",
    "Longitude",
    "EntryStreetName",
    "ExitStreetName",
    "EntryHeading",
    "ExitHeading",
    "Approach",
    "Hour",
    "Weekend",
    "Month",
    "HistoricalSeverity",
    "CongestionLevel",
    "TotalTimeStopped_p50",
    "TimeFromFirstStop_p50",
    "DistanceToFirstStop_p50",
]


metadata = (
    knowledge_base[
        metadata_columns
    ]
    .to_dict(
        orient="records"
    )
)


print_section(
    "VERIFYING IR METADATA"
)


required_fields = [
    "City",
    "IntersectionId",
    "Approach",
    "Hour",
    "Weekend",
    "Month",
    "HistoricalSeverity",
    "CongestionLevel",
]


for field in required_fields:

    if field not in metadata[0]:

        raise KeyError(
            f"Required IR metadata field "
            f"is missing: {field}"
        )

    print(
        f"  {field:<22}: OK"
    )


print_section(
    "BUILDING INVERTED INDEX"
)


inverted_index = InvertedIndex()

inverted_index.build(
    documents
)


statistics = (
    inverted_index
    .get_statistics()
)


print(
    "\nInverted index built successfully."
)


print(
    f"Indexed documents: "
    f"{statistics['indexed_documents']:,}"
)


print(
    f"Unique terms: "
    f"{statistics['unique_terms']:,}"
)


print(
    f"Total postings: "
    f"{statistics['total_postings']:,}"
)


inverted_index.print_sample(
    limit=25
)


print_section(
    "BUILDING TF-IDF RETRIEVER"
)


retriever = TrafficRetriever(
    documents=documents,
    metadata=metadata,
    inverted_index=inverted_index,
)


print_section(
    "SAVING IR MODEL"
)


print(
    "\nSaving IR model..."
)


with open(
    IR_MODEL_PATH,
    "wb",
) as file:

    pickle.dump(
        retriever,
        file,
    )


print(
    "\nIR model saved successfully to:"
)

print(
    IR_MODEL_PATH
)


test_query = (
    "atlanta intersection 84 "
    "weekday june hour 8 "
    "west approach severe congestion"
)


print_section(
    f"TEST QUERY: {test_query}"
)


results = retriever.search(
    query=test_query,
    top_k=5,
)


if not results:

    print(
        "\nNo results found."
    )

else:

    for rank, result in enumerate(
        results,
        start=1,
    ):

        data = result[
            "metadata"
        ]

        print(
            f"\nResult #{rank}"
        )

        print(
            f"Similarity: "
            f"{result['score']:.4f}"
        )

        print(
            f"City: "
            f"{data['City']}"
        )

        print(
            f"Intersection: "
            f"{data['IntersectionId']}"
        )

        print(
            f"Approach: "
            f"{data['Approach']}"
        )

        print(
            f"Hour: "
            f"{data['Hour']}"
        )

        print(
            f"Weekend: "
            f"{data['Weekend']}"
        )

        print(
            f"Month: "
            f"{data['Month']}"
        )

        print(
            f"Historical severity: "
            f"{data['HistoricalSeverity']:.4f}"
        )

        print(
            f"Congestion level: "
            f"{data['CongestionLevel']}"
        )

        print(
            f"Stopped time: "
            f"{data['TotalTimeStopped_p50']}"
        )

        print(
            f"Delay: "
            f"{data['TimeFromFirstStop_p50']}"
        )

        print(
            f"Distance: "
            f"{data['DistanceToFirstStop_p50']}"
        )


print_section(
    "IR BUILD COMPLETE"
)


print(
    "\nThe IR knowledge base now contains:"
)

print(
    "  1. Searchable historical traffic documents"
)

print(
    "  2. N/S/E/W approach metadata"
)

print(
    "  3. Numeric historical severity"
)

print(
    "  4. Congestion category labels"
)

print(
    "  5. City, intersection, hour, weekend, "
    "and month context"
)