"""ui/styles.py - global CSS for the dashboard's dark Traffic Operations Control Centre aesthetic; the one source of visual truth."""

from textwrap import dedent


# Centralized here so every other UI module references the same
# palette instead of hardcoding hex values in multiple places.
BACKGROUND = "#0a0e12"
PANEL = "#12181f"
PANEL_ALT = "#161d25"
BORDER = "#26313c"
TEXT = "#e4e9ee"
MUTED = "#7c8894"
ASPHALT = "#22282e"
LANE_MARKING = "#c9d1d8"

BRAND_ACCENT = "#2dd4bf"

SIGNAL_RED = "#c0392b"
SIGNAL_YELLOW = "#c9a227"
SIGNAL_GREEN = "#2f9e5c"
SIGNAL_OFF = "#2a2f34"

CONGESTION_LOW = "#2f9e5c"
CONGESTION_MODERATE = "#c9a227"
CONGESTION_HIGH = "#c0722b"
CONGESTION_SEVERE = "#c0392b"


def render_html(content):
    """
    Strip blank lines before handing raw HTML to st.markdown().

    CommonMark (the Markdown parser Streamlit runs HTML through,
    even with unsafe_allow_html=True) treats a <div>...</div> block
    as raw HTML only up to the first blank line - after that it
    falls back to an indented code block, showing literal tags as
    text. Blank lines carry no meaning in HTML, so removing them
    fixes this without changing how anything renders.
    """

    import streamlit as st

    lines = [
        line
        for line in content.splitlines()
        if line.strip() != ""
    ]

    st.markdown(
        "\n".join(lines),
        unsafe_allow_html=True,
    )


def get_global_css():

    return dedent(f"""
    <style>

    :root {{
        --background: {BACKGROUND};
        --panel: {PANEL};
        --panel-alt: {PANEL_ALT};
        --border: {BORDER};
        --text: {TEXT};
        --muted: {MUTED};
        --brand-accent: {BRAND_ACCENT};
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        background:
            radial-gradient(
                ellipse 1200px 600px at 15% -10%,
                rgba(45, 212, 191, 0.05),
                transparent 60%
            ),
            var(--background);
        color: var(--text);
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stMainBlockContainer"] {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1900px;
    }}

    h1, h2, h3, p, div, span, label {{
        font-family:
            "Segoe UI", -apple-system, BlinkMacSystemFont,
            Arial, sans-serif;
    }}

    .stc-header {{
        border: 1px solid var(--border);
        border-top: 2px solid var(--brand-accent);
        background: var(--panel);
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 0 24px rgba(45, 212, 191, 0.08);
    }}

    .stc-title {{
        color: #f2f5f7;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: 0.6px;
        margin: 0;
    }}

    .stc-tagline {{
        color: var(--brand-accent);
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.3px;
        text-transform: uppercase;
        margin-top: 4px;
    }}

    .stc-badge-row {{
        display: flex;
        gap: 10px;
        margin-top: 10px;
        flex-wrap: wrap;
    }}

    .stc-badge {{
        border: 1px solid rgba(45, 212, 191, 0.35);
        background: var(--panel-alt);
        color: var(--brand-accent);
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 10px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 4px 10px;
    }}

    .stc-panel {{
        border: 1px solid var(--border);
        border-top: 2px solid rgba(45, 212, 191, 0.45);
        background: var(--panel);
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.24);
    }}

    .stc-panel-title {{
        color: #cfd8de;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(45, 212, 191, 0.18);
        padding-bottom: 8px;
        margin-bottom: 10px;
    }}

    .stc-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 12px;
    }}

    .stc-row:last-child {{
        border-bottom: none;
    }}

    .stc-row-label {{
        color: var(--muted);
        letter-spacing: 0.3px;
    }}

    .stc-row-value {{
        color: var(--text);
        font-family: "SFMono-Regular", Consolas, monospace;
        font-weight: 600;
    }}

    .stc-heatmap-row {{
        padding: 8px 0;
    }}

    .stc-heatmap-label-row {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 6px;
    }}

    .stc-heatmap-label {{
        color: var(--text);
        font-size: 12px;
        font-family: "SFMono-Regular", Consolas, monospace;
        letter-spacing: 0.4px;
    }}

    .stc-heatmap-level {{
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }}

    .stc-heatmap-track {{
        height: 6px;
        background: var(--panel-alt);
        border: 1px solid var(--border);
        overflow: hidden;
    }}

    .stc-heatmap-fill {{
        height: 100%;
        transition: width 0.6s ease;
    }}

    table.stc-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }}

    table.stc-table th {{
        color: var(--muted);
        background: var(--panel-alt);
        border: 1px solid var(--border);
        font-size: 10px;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        padding: 7px 8px;
        text-align: left;
    }}

    table.stc-table td {{
        color: var(--text);
        background: var(--panel);
        border: 1px solid var(--border);
        padding: 7px 8px;
        font-family: "SFMono-Regular", Consolas, monospace;
    }}

    table.stc-table tr.stc-row-active td {{
        background: #17251d;
    }}

    div.stButton > button {{
        border-radius: 8px;
        border: 1px solid #384652;
        background: #171f27;
        color: #e7eef4;
        font-weight: 600;
        min-height: 42px;
        transition:
            border-color 0.15s ease,
            box-shadow 0.15s ease,
            transform 0.1s ease,
            background 0.15s ease;
    }}

    div.stButton > button:hover {{
        border-color: var(--brand-accent);
        background: #1c2830;
        color: white;
        box-shadow: 0 0 16px rgba(45, 212, 191, 0.35);
        transform: translateY(-1px);
    }}

    div.stButton > button:active {{
        transform: translateY(0);
        box-shadow: 0 0 8px rgba(45, 212, 191, 0.3);
    }}

    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {{
        background: linear-gradient(180deg, #34e0c9, #17b8a3);
        border: 1px solid #17b8a3;
        color: #06201c;
    }}

    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background: linear-gradient(180deg, #45e8d2, #1fcab3);
        border-color: #45e8d2;
        color: #06201c;
        box-shadow: 0 0 20px rgba(45, 212, 191, 0.55);
    }}

    .stc-time-readout {{
        border: 1px solid var(--border);
        background: var(--panel);
        min-height: 38px;
        padding: 4px 9px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        font-family: "SFMono-Regular", Consolas, monospace;
    }}

    .stc-time-readout span {{
        color: var(--muted);
        font-size: 8px;
        letter-spacing: 0.7px;
    }}

    .stc-time-readout strong {{
        color: var(--brand-accent);
        font-size: 13px;
        letter-spacing: 0.3px;
    }}

    [data-testid="stExpander"] {{
        border: 1px solid var(--border);
        background: var(--panel);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.24);
    }}

    [data-testid="stExpander"] summary {{
        color: var(--text);
    }}

    [data-testid="stExpander"] summary:hover {{
        color: var(--brand-accent);
    }}

    [data-testid="stSlider"] [role="slider"] {{
        box-shadow:
            0 0 0 4px rgba(45, 212, 191, 0.22),
            0 0 12px rgba(45, 212, 191, 0.45);
        transition: box-shadow 0.15s ease;
    }}

    [data-testid="stSlider"] [role="slider"]:hover,
    [data-testid="stSlider"] [role="slider"]:focus {{
        box-shadow:
            0 0 0 6px rgba(45, 212, 191, 0.28),
            0 0 18px rgba(45, 212, 191, 0.6);
    }}

    </style>
    """)
