from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Any
import httpx
import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "").rstrip("/")
MODEL_PATH = Path(__file__).parent / "data" / "artifacts" / "model.pkl"

if "live_history" not in st.session_state:
    st.session_state.live_history = []

st.set_page_config(page_title="MIND / Personal News", page_icon="✦", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner=False)
def load_artifact() -> dict[str, Any] | None:
    if not MODEL_PATH.is_file():
        return None
    with MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)


def infer_interest_terms(history: list[str], artifact: dict[str, Any] | None, limit: int = 8) -> list[str]:
    if not artifact or not history or "vectorizer" not in artifact:
        return []
    mapping = artifact["news_id_to_index"]
    indices = list(dict.fromkeys(mapping[item] for item in history if item in mapping))
    if not indices:
        return []
    matrix = csr_matrix(artifact["item_matrix"], dtype=np.float32)
    profile = np.asarray(matrix[indices].sum(axis=0)).ravel()
    names = np.asarray(artifact["vectorizer"].get_feature_names_out())
    return names[np.argsort(-profile)[:limit]].tolist()


def clicked_titles(history: list[str], artifact: dict[str, Any] | None) -> list[tuple[str, str]]:
    if not artifact:
        return []
    mapping = artifact["news_id_to_index"]
    news = artifact["news"]
    return [
        (item, str(news.iloc[mapping[item]]["title"]))
        for item in dict.fromkeys(history)
        if item in mapping
    ]
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#32205a; --muted:#7e7098; --purple:#9a35e8; --violet:#6f24d9; --soft:#f7ecff; --line:#eadcf5; }
    .stApp { background: linear-gradient(135deg,#fbfaff 0%,#f6effb 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp div, .stApp li { color:var(--ink); }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1240px; padding:2.5rem 2.5rem 3rem; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif; color:var(--ink); }
    .topbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:2.4rem; }
    .brand { font:700 1.15rem 'Space Grotesk'; letter-spacing:.03em; color:var(--ink); }
    .brand span { color:var(--purple); }
    .eyebrow { color:var(--purple); font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
    .hero { background:linear-gradient(115deg,#7926dc,#b443ef); border-radius:26px; padding:2.5rem 2.8rem; color:white; box-shadow:0 18px 45px rgba(126,41,213,.22); min-height:220px; }
    .hero h1 { color:white; font-size:2.75rem; line-height:1.04; margin:.45rem 0 .75rem; }
    .hero p { color:#f3ddff; max-width:560px; font-size:1.05rem; margin:0; }
    .hero-pill { display:inline-block; background:rgba(255,255,255,.17); border:1px solid rgba(255,255,255,.22); padding:.45rem .8rem; border-radius:100px; font-size:.76rem; }
    .panel { background:rgba(255,255,255,.78); border:1px solid var(--line); border-radius:22px; padding:1.35rem 1.45rem; box-shadow:0 12px 30px rgba(75,43,104,.06); }
    .result-card { background:white; border:1px solid var(--line); border-radius:15px; padding:.8rem 1rem; margin:.55rem 0; transition:.15s ease; }
    .result-card:hover { border-color:#c783f4; transform:translateX(3px); }
    .rank { color:var(--purple); font:700 .8rem 'Space Grotesk'; margin-right:.8rem; }
    .metric { background:var(--soft); border-radius:16px; padding:.9rem 1rem; text-align:center; }
    .metric strong { display:block; color:var(--violet); font:700 1.3rem 'Space Grotesk'; }
    .metric small { color:var(--muted); }
    [data-testid="stButton"] button { background:linear-gradient(100deg,#8c2fe3,#b43ff0); color:white; border:0; border-radius:12px; font-weight:700; padding:.7rem 1rem; box-shadow:0 8px 18px rgba(135,42,219,.2); }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { border:1px solid #dec9ed; border-radius:12px; background:white; }
    .caption { color:var(--muted) !important; font-size:.82rem; }
    [data-testid="stTextInput"] label, [data-testid="stTextArea"] label, [data-testid="stSlider"] label { color:var(--ink) !important; font-weight:600; }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { color:var(--ink) !important; caret-color:var(--purple); }
    [data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder { color:#9686a8 !important; opacity:1; }
    [data-testid="stAlert"] p { color:var(--ink) !important; }
    [data-testid="stAlert"] { border-radius:12px; }
    [data-testid="stSlider"] [role="slider"] { background:#8c2fe3; }
    .stButton button p { color:white !important; }
    .article-title { color:var(--ink); font-weight:700; font-size:1rem; }
    .article-meta { color:var(--muted); font-size:.78rem; margin-top:.2rem; }
    .article-reason { color:#5b4777; font-size:.82rem; margin-top:.45rem; }
    </style>
    <div class="topbar"><div class="brand">MIND <span>✦</span> NEWS LAB</div><div class="caption">CONTENT-BASED RECOMMENDATION</div></div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><span class="hero-pill">TF-IDF + COSINE SIMILARITY</span><h1>Find your next<br>great read.</h1><p>Build a reading profile from clicked MIND news IDs and receive fast, explainable recommendations.</p></div>',
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([0.86, 1.35], gap="large")
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">01 / Reading profile</div>', unsafe_allow_html=True)
    st.subheader("Tell us what you read")
    user_id = st.text_input("User ID", value="demo-user", label_visibility="collapsed", placeholder="User ID")
    history_text = st.text_area(
        "Clicked news IDs", label_visibility="collapsed", placeholder="Paste news IDs, separated by commas or new lines…", height=150,
    )
    top_k = st.slider("Number of recommendations", 1, 20, 8)
    submitted = st.button("Generate recommendations  →", use_container_width=True)
    st.markdown('<p class="caption">Unknown IDs are safely ignored. Empty history uses a cold-start fallback.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

current_history = [
    item.strip()
    for item in history_text.replace("\n", ",").split(",")
    if item.strip()
]
if submitted:
    st.session_state.live_history = list(dict.fromkeys(current_history))

live_history = st.session_state.live_history
artifact = load_artifact()
interest_terms = infer_interest_terms(live_history, artifact)
title_rows = clicked_titles(live_history, artifact)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">02 / Your results</div>', unsafe_allow_html=True)
    st.subheader("A shortlist made for you")
    if submitted:
        payload: dict[str, Any] = {"user_id": user_id.strip(), "click_history": live_history, "top_k": top_k}
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(f"{API_URL}/recommend", json=payload)
                response.raise_for_status()
            result = response.json()
            st.success("Recommendation profile updated")
            for rank, item in enumerate(result.get("recommendations", []), start=1):
                score = float(item.get("similarity_score", 0.0)) * 100
                st.markdown(
                    f'<div class="result-card"><span class="rank">{rank:02d}</span>'
                    f'<span class="article-title">{item["title"]}</span>'
                    f'<div class="article-meta">{item["news_id"]} · {item["category"]} · {score:.1f}% content similarity</div>'
                    f'<div class="article-reason">Why this is recommended: {item["reason"]}</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown(f'<div class="metric"><strong>{result["latency_ms"]:.2f} ms</strong><small>inference latency · persisted to PostgreSQL</small></div>', unsafe_allow_html=True)
        except httpx.HTTPStatusError as exc:
            st.error(f"API returned {exc.response.status_code}: {exc.response.text}")
        except httpx.RequestError:
            st.error(f"Could not reach `{API_URL}`. Start FastAPI first.")
        except (KeyError, ValueError) as exc:
            st.error(f"Unexpected API response: {exc}")
    else:
        st.info("Add a few clicked news IDs, then generate your shortlist.")
    if live_history:
        st.markdown('<div class="eyebrow">03 / Live interest map</div>', unsafe_allow_html=True)
        st.markdown(f"**{len(live_history)} clicked article(s)** are shaping this session profile.")
        if interest_terms:
            st.markdown("**Inferred interest terms**")
            st.markdown(" · ".join(f"`{term}`" for term in interest_terms))
        if title_rows:
            with st.expander("View articles contributing to the profile"):
                for news_id, title in title_rows:
                    st.markdown(f"**{news_id}** — {title}")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.markdown('<div class="caption">FastAPI backend · PostgreSQL telemetry · MINDlarge training data · Explainable item similarity</div>', unsafe_allow_html=True)
