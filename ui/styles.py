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
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        background: var(--background);
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
        background: var(--panel);
        padding: 16px 20px;
        margin-bottom: 12px;
    }}

    .stc-title {{
        color: #f2f5f7;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: 0.6px;
        margin: 0;
    }}

    .stc-subtitle {{
        color: var(--muted);
        font-size: 11px;
        letter-spacing: 1.3px;
        text-transform: uppercase;
        margin-top: 6px;
    }}

    .stc-badge-row {{
        display: flex;
        gap: 10px;
        margin-top: 10px;
        flex-wrap: wrap;
    }}

    .stc-badge {{
        border: 1px solid var(--border);
        background: var(--panel-alt);
        color: var(--muted);
        font-size: 10px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 4px 10px;
    }}

    .stc-panel {{
        border: 1px solid var(--border);
        background: var(--panel);
        padding: 14px 16px;
        margin-bottom: 12px;
    }}

    .stc-panel-title {{
        color: #cfd8de;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        border-bottom: 1px solid var(--border);
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

    .stc-value-lg {{
        font-size: 22px;
        font-weight: 700;
        color: var(--text);
        font-family: "SFMono-Regular", Consolas, monospace;
    }}

    .stc-value-label {{
        color: var(--muted);
        font-size: 10px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-top: 2px;
    }}

    .stc-grid-2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1px;
        background: var(--border);
        border: 1px solid var(--border);
    }}

    .stc-grid-cell {{
        background: var(--panel);
        padding: 10px 12px;
    }}

    .stc-tag {{
        display: inline-block;
        padding: 2px 8px;
        font-size: 10px;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        border: 1px solid var(--border);
    }}

    .stc-tag-switch {{ color: #e0866f; border-color: #6b3a2f; }}
    .stc-tag-hold {{ color: #6fb98c; border-color: #2f5c40; }}
    .stc-tag-transition {{ color: #c9a227; border-color: #5c4e1f; }}
    .stc-tag-activate {{ color: #7ab0d8; border-color: #2f4a5c; }}
    .stc-tag-override {{ color: #d84f4f; border-color: #6b2f2f; }}

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
        border-radius: 2px;
        border: 1px solid #384652;
        background: #171f27;
        color: #e7eef4;
        font-weight: 600;
        min-height: 38px;
    }}

    div.stButton > button:hover {{
        border-color: #5c7285;
        background: #1c2830;
        color: white;
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
        color: var(--text);
        font-size: 13px;
        letter-spacing: 0.3px;
    }}

    [data-testid="stTabs"] button {{
        border-radius: 0;
        color: var(--muted);
    }}

    [data-testid="stTabs"] button[aria-selected="true"] {{
        color: var(--text);
        border-bottom-color: #5c8ca8;
    }}

    </style>
    """)
