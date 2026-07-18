from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

ENV_PATH = (
    PROJECT_ROOT
    / ".env"
)

load_dotenv(
    dotenv_path=ENV_PATH
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = "gemini-3.5-flash"

# Operator-facing explanations are a short status note, not a report -
# the prompt itself asks for 2-3 sentences (~60 words); this cap is a
# backstop, not the primary length control. gemini-3.5-flash is a
# Gemini 3.x model, where thinking CANNOT be fully disabled (unlike
# 2.5 Flash's thinking_budget=0) - thinking_level=MINIMAL below is
# the lowest available setting, but it still consumes part of this
# budget before the visible answer. 400 was too tight: real responses
# were getting cut off mid-sentence once thinking exhausted it.
MAX_OUTPUT_TOKENS = 1024


def create_gemini_client():
    """
    Create and return the Gemini API client.
    """

    if not GEMINI_API_KEY:

        raise ValueError(
            "\nGEMINI_API_KEY was not found.\n\n"
            "Create a .env file in the project root "
            "and add:\n\n"
            "GEMINI_API_KEY=your_api_key_here"
        )

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    return client


def build_historical_context(
    approach,
    ml_class,
    ml_severity,
    ir_severity,
    retrieved_cases,
):
    """
    Convert retrieved historical traffic cases into
    grounded textual context for Gemini.
    """

    context_parts = []

    context_parts.append(
        "CURRENT TRAFFIC ANALYSIS"
    )

    context_parts.append(
        f"Approach: {approach}"
    )

    context_parts.append(
        f"ML congestion class: {ml_class}"
    )

    context_parts.append(
        f"ML severity score: {ml_severity:.4f}"
    )

    context_parts.append(
        f"IR historical severity score: "
        f"{ir_severity:.4f}"
    )

    context_parts.append(
        "\nRETRIEVED HISTORICAL TRAFFIC CASES"
    )

    if not retrieved_cases:

        context_parts.append(
            "No historical cases were retrieved."
        )

    else:

        for rank, result in enumerate(
            retrieved_cases,
            start=1,
        ):

            metadata = result.get(
                "metadata",
                {}
            )

            document = result.get(
                "document",
                "Unknown"
            )

            similarity = float(
                result.get(
                    "score",
                    0.0
                )
            )

            context_parts.append(
                (
                    f"\nHistorical Case #{rank}\n"
                    f"Similarity: {similarity:.4f}\n"
                    f"City: "
                    f"{metadata.get('City', 'Unknown')}\n"
                    f"Intersection: "
                    f"{metadata.get('IntersectionId', 'Unknown')}\n"
                    f"Hour: "
                    f"{metadata.get('Hour', 'Unknown')}\n"
                    f"Weekend: "
                    f"{metadata.get('Weekend', 'Unknown')}\n"
                    f"Month: "
                    f"{metadata.get('Month', 'Unknown')}\n"
                    f"Traffic record: {document}"
                )
            )

    return "\n".join(
        context_parts
    )


def build_rag_prompt(
    historical_context,
):
    """
    Build the grounded prompt sent to Gemini.

    Asks for a short, plain-text status note (2-3 sentences) rather
    than a multi-section report - this is operator-facing UI text,
    read at a glance next to a live signal, not a written analysis.
    MAX_OUTPUT_TOKENS backs this up as a hard cap independent of
    whether the model follows the length instruction exactly.
    """

    prompt = f"""
You are the explanation layer of an adaptive traffic-signal
controller, writing a short status note for a human operator to
read at a glance - not a report.

Using ONLY the evidence below, write EXACTLY 2-3 sentences (no more
than 60 words total) covering: the current congestion level,
whether the ML prediction and historical evidence agree or
disagree, and why this approach's signal priority is what it is.

Do not invent traffic measurements or historical cases - use only
what's in the evidence. Do not use headers, bullet points, numbered
lists, or markdown formatting of any kind - plain prose only, like
a status note, not an essay.

EVIDENCE:

{historical_context}

Now write the 2-3 sentence status note.
"""

    return prompt.strip()


def generate_traffic_explanation(
    approach,
    ml_class,
    ml_severity,
    ir_severity,
    retrieved_cases,
):
    """
    Generate a grounded RAG explanation using Gemini.
    """

    historical_context = (
        build_historical_context(
            approach=approach,
            ml_class=ml_class,
            ml_severity=ml_severity,
            ir_severity=ir_severity,
            retrieved_cases=retrieved_cases,
        )
    )

    prompt = build_rag_prompt(
        historical_context=historical_context
    )

    client = create_gemini_client()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
            ),
        ),
    )

    if not response.text:

        return (
            "Gemini returned an empty response."
        )

    return response.text.strip()