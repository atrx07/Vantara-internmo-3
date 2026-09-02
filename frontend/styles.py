"""Small, maintainable visual system for the Vantara business dashboard."""

from __future__ import annotations

import streamlit as st


def apply_dashboard_style() -> None:
    """Apply the dashboard's local CSS without external fonts or assets."""
    st.markdown(
        """
        <style>
        :root {
          --vantara-ink: #182528;
          --vantara-navy: #183238;
          --vantara-teal: #0f766e;
          --vantara-sand: #e7e3da;
          --vantara-paper: #f5f3ee;
          --vantara-coral: #d9785f;
        }
        .stApp { background: var(--vantara-paper); color: var(--vantara-ink); }
        .block-container { max-width: 1320px; padding-top: 1.4rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] { background: var(--vantara-navy); }
        [data-testid="stSidebar"] * { color: #f7f5f0; }
        [data-testid="stSidebar"] [role="radiogroup"] label {
          border-radius: 8px; padding: .3rem .45rem;
        }
        [data-testid="stMetric"] {
          background: rgba(255,255,255,.68); border: 1px solid rgba(24,50,56,.12);
          border-radius: 12px; padding: 1rem 1.1rem; min-height: 112px;
        }
        [data-testid="stMetricLabel"] { color: #5d6b6e; }
        [data-testid="stMetricValue"] { color: var(--vantara-navy); }
        .vantara-hero {
          border-top: 5px solid var(--vantara-teal); border-bottom: 1px solid var(--vantara-sand);
          padding: 1.1rem 0 1.25rem; margin: 0 0 1.5rem;
        }
        .vantara-eyebrow {
          color: var(--vantara-teal); font-size: .74rem; font-weight: 700;
          letter-spacing: .12em; text-transform: uppercase; margin-bottom: .35rem;
        }
        .vantara-hero h1 {
          color: var(--vantara-navy); font-family: Georgia, 'Times New Roman', serif;
          font-size: clamp(2.1rem, 4.2vw, 4.6rem); line-height: .98;
          letter-spacing: -.04em; max-width: 1000px; margin: 0;
        }
        .vantara-hero p { color: #5d6b6e; max-width: 760px; margin: .8rem 0 0; }
        .vantara-kicker { color: #5d6b6e; font-size: .85rem; }
        h2, h3 { color: var(--vantara-navy); letter-spacing: -.02em; }
        div[data-testid="stDataFrame"] {
          border: 1px solid rgba(24,50,56,.12); border-radius: 10px;
        }
        .stDownloadButton button, .stButton button[kind="primary"] {
          border-radius: 8px; font-weight: 650;
        }
        [data-testid="stAlert"] { border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, description: str, *, eyebrow: str) -> None:
    """Render consistent editorial page hierarchy."""
    st.markdown(
        f"""
        <div class="vantara-hero">
          <div class="vantara-eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
