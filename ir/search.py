from pathlib import Path
import pickle


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

IR_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ir"
    / "traffic_ir.pkl"
)


print("SMART TRAFFIC AI — HISTORICAL TRAFFIC SEARCH")


if not IR_MODEL_PATH.exists():

    raise FileNotFoundError(
        "\nIR model not found.\n"
        "Run this command first:\n\n"
        "python3 -m ir.build_index"
    )


print("\nLoading IR model...")


with open(
    IR_MODEL_PATH,
    "rb"
) as file:

    retriever = pickle.load(file)


print("IR model loaded successfully.")


while True:

    query = input(
        "\nEnter traffic search query "
        "(or 'exit' to quit):\n> "
    ).strip()


    if query.lower() in {
        "exit",
        "quit",
        "q"
    }:

        print(
            "\nExiting traffic search."
        )

        break


    if not query:

        print(
            "\nPlease enter a search query."
        )

        continue


    results = retriever.search(
        query,
        top_k=5
    )


    if not results:

        print(
            "\nNo matching historical "
            "traffic cases found."
        )

        continue


    print(
        f"\nTop {len(results)} "
        f"historical traffic cases:"
    )


    for rank, result in enumerate(
        results,
        start=1
    ):

        data = result["metadata"]


        print(
            f"\nRESULT #{rank}"
        )

        print(
            f"Similarity score: "
            f"{result['score']:.4f}"
        )

        print(
            f"City: "
            f"{data['City']}"
        )

        print(
            f"Intersection ID: "
            f"{data['IntersectionId']}"
        )

        print(
            f"Entry heading: "
            f"{data['EntryHeading']}"
        )

        print(
            f"Exit heading: "
            f"{data['ExitHeading']}"
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
            f"Stopped time p50: "
            f"{data['TotalTimeStopped_p50']:.2f} sec"
        )

        print(
            f"Delay p50: "
            f"{data['TimeFromFirstStop_p50']:.2f} sec"
        )

        print(
            f"Distance p50: "
            f"{data['DistanceToFirstStop_p50']:.2f}"
        )

        print(
            "\nIndexed document:"
        )

        print(
            result["document"]
        )


print(
    "\nSearch session complete."
)