"""ui/panels.py - panels for ML predictions, model performance metrics, and the Gemini/RAG explanation."""

import streamlit as st

from simulation.movement_definitions import (
    PHASES,
    TRAVEL_DIRECTIONS,
)

from ui.styles import render_html


def render_ml_panel(state):

    rows = "".join(
        f"""
        <div class="stc-row">
            <span class="stc-row-label">
                {direction} \u2014
                {state['ml_predictions'][direction]['class']}
            </span>
            <span class="stc-row-value">
                {state['ml_predictions'][direction]['severity']:.4f}
            </span>
        </div>
        """
        for direction in TRAVEL_DIRECTIONS
    )

    render_html(f"""
    <div class="stc-panel">
        <div class="stc-panel-title">
            ML Congestion Prediction (Approach-Level)
        </div>
        {rows}
    </div>
    """)


def render_model_performance_panel(model_bundle):

    selected_scope = model_bundle["model_scope"]

    comparisons = [
        ("Global model (all cities)", model_bundle["global_metrics"], False),
        ("Global model on Intersection 84", model_bundle["global_on_local_metrics"], selected_scope == "global"),
        ("Local Intersection 84 model", model_bundle["local_metrics"], selected_scope == "local"),
    ]

    rows = "".join(
        f"""
        <tr class="{'stc-row-active' if is_selected else ''}">
            <td>{label}{' — SELECTED' if is_selected else ''}</td>
            <td>{metrics['accuracy'] * 100:.2f}%</td>
            <td>{metrics['balanced_accuracy'] * 100:.2f}%</td>
            <td>{metrics['macro_precision']:.4f}</td>
            <td>{metrics['macro_recall']:.4f}</td>
            <td>{metrics['macro_f1']:.4f}</td>
            <td>{metrics['weighted_f1']:.4f}</td>
        </tr>
        """
        for label, metrics, is_selected in comparisons
    )

    render_html(f"""
    <div class="stc-panel">
        <div class="stc-panel-title">
            Model Performance — test month {model_bundle['test_month']} held out, never seen during training
        </div>
        <table class="stc-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Balanced Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Macro F1</th>
                    <th>Weighted F1</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """)


def render_rag_explanation_panel(state):
    """
    One Gemini explanation per approach, grounded in the same ML
    prediction + IR retrieved cases used elsewhere. Text is
    regenerated only when the simulated hour advances (see
    AdaptiveTrafficSimulation._build_rag_explanations), so this
    reads state['rag_explanations'] rather than calling the API.
    """

    render_html("""
    <div class="stc-panel">
        <div class="stc-panel-title" style="text-align: center; font-size: 13px;">
            AI Traffic Explanation — Gemini, Grounded on ML + IR
        </div>
    </div>
    """)

    grid_rows = [
        TRAVEL_DIRECTIONS[0:2],
        TRAVEL_DIRECTIONS[2:4],
    ]

    for row_directions in grid_rows:

        columns = st.columns(len(row_directions), gap="medium")

        for column, direction in zip(columns, row_directions):

            is_current_approach = (
                f"{direction}_STRAIGHT" in PHASES[state["current_phase"]]
                or f"{direction}_LEFT" in PHASES[state["current_phase"]]
                or f"{direction}_RIGHT" in PHASES[state["current_phase"]]
            )

            with column:
                with st.expander(
                    f"{direction} — grounded explanation",
                    expanded=is_current_approach,
                ):
                    st.markdown(state["rag_explanations"][direction])
