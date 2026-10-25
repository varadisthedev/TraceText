import html
import re
import sys
import warnings

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --- palette ---------------------------------------------------------------
RED = "#ED1C24"      # Racing Red — AI
BLUE = "#235789"     # Baltic Blue — brand / headers
GOLD = "#F1D302"     # Bright Gold — accents
INK = "#020100"
GREEN = "#1E8E3E"    # Human (not in the brand palette; added for red/green AI-vs-human signal)
MUTED = "#6B7280"

MODEL_PATH = "models/tracetext_model.joblib"
BANNER_PATH = "assets/ai_or_human_banner_900x300(1).png"
GITHUB_URL = "https://github.com/varadisthedev/TraceText"
DATASET_URL = "https://huggingface.co/datasets/liamdugan/raid"

st.set_page_config(page_title="TraceText", page_icon=":material/fingerprint:", layout="wide")

# --- model -------------------------------------------------------------
@st.cache_resource
def load_model():
    # models/tracetext_model.joblib is first-party, committed by the project
    # owner (not user-uploaded), so loading this pickle is trusted.
    # The pickled SGDClassifier's Cython loss object reports its module as the
    # bare name "_loss" instead of "sklearn._loss.loss" (a known sklearn
    # pickling quirk from the Colab build). Alias it so joblib can resolve it.
    from sklearn._loss import loss as _loss_mod
    sys.modules["_loss"] = _loss_mod
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(MODEL_PATH)


model = load_model()
AI_CLASS_INDEX = list(model.classes_).index(1)  # label convention: 0 = human, 1 = AI

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


MIN_WORDS_PER_CHUNK = 40  # TF-IDF on shorter chunks is too sparse/noisy to score reliably


def chunk_text(text: str, max_chunks: int = 10) -> list[str]:
    """Group sentences into up to `max_chunks` roughly equal chunks, each with
    at least MIN_WORDS_PER_CHUNK words so short inputs aren't over-split into
    noisy single-sentence chunks."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    word_count = len(text.split())
    n_chunks = max(1, min(max_chunks, len(sentences), word_count // MIN_WORDS_PER_CHUNK))
    chunk_size = -(-len(sentences) // n_chunks)  # ceil div
    return [
        " ".join(sentences[i : i + chunk_size])
        for i in range(0, len(sentences), chunk_size)
    ]


def analyze(text: str):
    chunks = chunk_text(text)
    probs = model.predict_proba(chunks)[:, AI_CLASS_INDEX]  # P(AI) per chunk
    weights = np.array([max(len(c.split()), 1) for c in chunks], dtype=float)
    ai_pct = float(np.average(probs, weights=weights) * 100)
    sentences = split_sentences(text)
    words = text.split()
    return {
        "chunks": chunks,
        "probs": probs,
        "ai_pct": ai_pct,
        "human_pct": 100 - ai_pct,
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_len": len(words) / len(sentences) if sentences else 0,
    }


# --- samples -----------------------------------------------------------
HUMAN_SAMPLE = (
    "honestly I didn't think I'd finish this book but here we are, 2am and "
    "still reading. my sister told me the ending was sad so I've been bracing "
    "myself for like three chapters now lol. anyway I should probably sleep, "
    "work is going to be rough tomorrow if I don't."
)
AI_SAMPLE = (
    "Artificial intelligence has fundamentally transformed the way organizations "
    "approach problem-solving. Furthermore, these systems are capable of processing "
    "vast amounts of information in a fraction of the time required by traditional "
    "methods. In conclusion, continued research into responsible AI development "
    "remains essential to fully realize these benefits."
)
MIXED_SAMPLE = (
    HUMAN_SAMPLE
    + " "
    + AI_SAMPLE
    + " The weather here has been unusually warm this week and I keep forgetting "
    "to bring a jacket when I leave in the morning, which is annoying because it "
    "always cools down by evening. Moreover, the proposed framework demonstrates "
    "significant improvements over baseline approaches across all evaluated metrics. "
    "Overall, these results highlight the importance of robust evaluation methodology "
    "in natural language processing research."
)

if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "result" not in st.session_state:
    st.session_state.result = None


def set_sample(text: str):
    st.session_state.text_input = text
    st.session_state.result = None


# --- style ---------------------------------------------------------------
st.markdown(
    f"""
    <style>
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1160px; }}

    .stApp {{
        background: linear-gradient(180deg, #DCE4F0 0%, #EDF1F7 340px, #EDF1F7 100%);
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 16px !important;
        background: #FFFFFF !important;
        border: 1px solid #DDE3EE !important;
        box-shadow: 0 2px 10px rgba(2, 23, 59, 0.06);
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {{
        gap: 0.65rem;
    }}
    [data-testid="stImage"] img {{
        border-radius: 0;
    }}

    .tt-icon {{
        font-family: 'Material Symbols Rounded';
        font-weight: 400;
        font-style: normal;
        font-feature-settings: 'liga';
        -webkit-font-feature-settings: 'liga';
        -moz-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
        white-space: nowrap;
        vertical-align: middle;
    }}
    .tt-card-head {{ display: flex; align-items: center; gap: 0.6rem; }}
    .tt-icon-badge {{
        width: 34px; height: 34px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.15rem; line-height: 1; color: white; flex-shrink: 0;
    }}
    .tt-card-title {{ font-weight: 700; font-size: 1.05rem; color: {INK}; }}
    .tt-card-sub {{ color: {MUTED}; font-size: 0.85rem; margin: 0.15rem 0 0 44px; }}

    .stat-box {{
        background: #EEF2F8;
        border: 1px solid #DEE5F0;
        border-radius: 10px;
        padding: 0.65rem 0.8rem;
        text-align: center;
    }}
    .stat-box .v {{ font-size: 1.25rem; font-weight: 700; color: {INK}; }}
    .stat-box .l {{ font-size: 0.74rem; color: {MUTED}; }}

    .verdict-pill {{
        display: inline-flex; align-items: center; gap: 0.45rem;
        padding: 0.4rem 1rem; border-radius: 999px; font-weight: 700;
        font-size: 1.05rem; color: white;
    }}
    .confidence-value {{ font-size: 1.7rem; font-weight: 800; line-height: 1; color: {INK}; }}
    .confidence-label {{ color: {MUTED}; font-size: 0.78rem; }}

    .chunk-list {{ max-height: 430px; overflow-y: auto; padding-right: 0.3rem; }}
    .chunk-span {{
        padding: 0.45rem 0.65rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        display: block;
        line-height: 1.5;
        font-size: 0.92rem;
        border-left: 4px solid transparent;
    }}
    .chunk-tag {{
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.08rem 0.5rem;
        border-radius: 999px;
        color: white;
        margin-right: 0.45rem;
        white-space: nowrap;
    }}

    .empty-state {{
        text-align: center; padding: 2.6rem 1rem; color: #9AA2AE;
    }}
    .empty-state .ico {{ font-size: 2.1rem; margin-bottom: 0.4rem; }}

    .stButton > button {{ border-radius: 10px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

GITHUB_SVG = """<svg height="16" width="16" viewBox="0 0 16 16" fill="currentColor">
<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53
.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0
1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0
3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01
8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>"""


def card_header(icon: str, title: str, subtitle: str | None = None, accent: str = BLUE):
    st.markdown(
        f'<div class="tt-card-head"><span class="tt-icon-badge" style="background:{accent};">'
        f'<span class="tt-icon">{icon}</span></span><span class="tt-card-title">{title}</span></div>'
        + (f'<div class="tt-card-sub">{subtitle}</div>' if subtitle else ""),
        unsafe_allow_html=True,
    )


def stat_row(items, cols=3):
    boxes = st.columns(cols)
    for box, (val, label) in zip(boxes, items):
        box.markdown(
            f'<div class="stat-box"><div class="v">{val}</div><div class="l">{label}</div></div>',
            unsafe_allow_html=True,
        )


# --- hero --------------------------------------------------------------
st.image(BANNER_PATH, width="stretch")

st.write("")

# --- input / result ------------------------------------------------------
col_in, col_out = st.columns([1, 1], gap="medium")

with col_in:
    with st.container(border=True):
        card_header("description", "Enter Text", "Paste your text below, or load a sample to try the model.")
        text = st.text_area(
            "input", key="text_input", height=190, label_visibility="collapsed",
            placeholder="Paste an article, essay, or any passage of text here...",
        )
        st.caption(f"{len(text.split())} words")

        s1, s2, s3 = st.columns(3)
        s1.button("Human", icon=":material/person:", on_click=set_sample, args=(HUMAN_SAMPLE,), width="stretch")
        s2.button("AI", icon=":material/smart_toy:", on_click=set_sample, args=(AI_SAMPLE,), width="stretch")
        s3.button("Mixed", icon=":material/shuffle:", on_click=set_sample, args=(MIXED_SAMPLE,), width="stretch")
        predict_clicked = st.button(
            "Predict", icon=":material/arrow_forward:", type="primary", width="stretch"
        )

        if predict_clicked:
            if text.strip():
                st.session_state.result = analyze(text)
            else:
                st.session_state.result = None
                st.warning("Enter some text first.")

with col_out:
    with st.container(border=True):
        card_header("auto_awesome", "Prediction Result", accent=RED)
        result = st.session_state.result

        if not result:
            st.markdown(
                '<div class="empty-state"><div class="ico"><span class="tt-icon">query_stats</span></div>'
                "Run a prediction to see the AI / Human breakdown here.</div>",
                unsafe_allow_html=True,
            )
        else:
            ai_pct, human_pct = result["ai_pct"], result["human_pct"]
            verdict_ai = ai_pct >= 50
            verdict_color = RED if verdict_ai else GREEN
            verdict_label = "AI Generated" if verdict_ai else "Human Written"
            verdict_icon = "smart_toy" if verdict_ai else "person"
            confidence = max(ai_pct, human_pct)

            st.write("")
            c1, c2 = st.columns([1, 1], vertical_alignment="center")
            c1.markdown(
                f'<span class="verdict-pill" style="background:{verdict_color};">'
                f'<span class="tt-icon">{verdict_icon}</span> {verdict_label}</span>',
                unsafe_allow_html=True,
            )
            c2.markdown(
                f'<div style="text-align:right;"><div class="confidence-value">{confidence:.1f}%</div>'
                f'<div class="confidence-label">ensembled confidence</div></div>',
                unsafe_allow_html=True,
            )

            st.write("")
            stat_row(
                [
                    (result["word_count"], "Word Count"),
                    (result["sentence_count"], "Sentence Count"),
                    (f"{result['avg_sentence_len']:.1f}", "Avg Sentence Length"),
                ]
            )

            st.write("")
            donut_col, legend_col = st.columns([1, 1], vertical_alignment="center")
            with donut_col:
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=["AI", "Human"],
                            values=[ai_pct, human_pct],
                            hole=0.68,
                            marker_colors=[RED, GREEN],
                            textinfo="none",
                            sort=False,
                        )
                    ]
                )
                fig.update_layout(
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=160,
                    annotations=[
                        dict(text=f"{ai_pct:.0f}%<br>AI", x=0.5, y=0.5, font_size=17, showarrow=False)
                    ],
                )
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            with legend_col:
                n_ai_chunks = int(np.sum(result["probs"] >= 0.5))
                n_chunks = len(result["chunks"])
                st.markdown(
                    f"""
                    <div><span style="color:{RED};">●</span> AI — {ai_pct:.1f}%</div>
                    <div><span style="color:{GREEN};">●</span> Human — {human_pct:.1f}%</div>
                    <div style="color:{MUTED};font-size:0.8rem;margin-top:0.35rem;">
                    {n_ai_chunks} of {n_chunks} chunks scored AI-like</div>
                    """,
                    unsafe_allow_html=True,
                )

            st.info(
                f"Scored **{ai_pct:.1f}% AI / {human_pct:.1f}% human**, averaged across chunks "
                "and weighted by chunk length. This is a probability from a TF-IDF + "
                "logistic-regression baseline — a signal, not a verdict."
            )

# --- chunk breakdown -----------------------------------------------------
if st.session_state.result:
    result = st.session_state.result
    st.write("")
    with st.container(border=True):
        card_header(
            "grid_view", "Chunk-by-Chunk Breakdown",
            "Up to 10 chunks, each scored independently and then ensembled into the result above.",
        )
        st.write("")
        rows = []
        for i, (chunk, prob) in enumerate(zip(result["chunks"], result["probs"]), start=1):
            is_ai = prob >= 0.5
            color = RED if is_ai else GREEN
            tint = "#FDECEC" if is_ai else "#E8F5EA"
            tag = f"AI {prob * 100:.0f}%" if is_ai else f"HUMAN {(1 - prob) * 100:.0f}%"
            rows.append(
                f'<div class="chunk-span" style="background:{tint};border-left-color:{color};">'
                f'<span class="chunk-tag" style="background:{color};">chunk {i} · {tag}</span>'
                f"{html.escape(chunk)}</div>"
            )
        st.markdown(f'<div class="chunk-list">{"".join(rows)}</div>', unsafe_allow_html=True)

st.write("")

# --- model performance (real numbers from the training notebook's grouped
# 5-fold CV and held-out test split — notebook/TraceText_MLDS_Final_Improved.ipynb) ---
CV_COMPARISON = [
    ("TF-IDF + SGD Classifier", 0.899, 0.914, 0.881, 0.897),
    ("TF-IDF + Logistic Regression", 0.883, 0.877, 0.890, 0.883),
    ("TF-IDF + Multinomial NB", 0.736, 0.672, 0.917, 0.775),
]
TEST_REPORT = {
    "Human": dict(precision=0.88, recall=0.90, f1=0.89, support=403),
    "AI": dict(precision=0.89, recall=0.88, f1=0.89, support=405),
}
TEST_ACCURACY = 0.89

perf_col, info_col = st.columns([1.4, 1], gap="medium")

with perf_col:
    with st.container(border=True):
        card_header(
            "bar_chart", "Model Performance",
            "Grouped 5-fold CV used to pick the deployed classifier, plus its score on a held-out "
            "split it never trained on (grouped by source, so related passages can't leak across).",
        )
        st.write("")
        names = [r[0].replace("TF-IDF + ", "") for r in CV_COMPARISON]
        fig = go.Figure()
        for idx, (metric, color) in enumerate(
            [("Accuracy", BLUE), ("Precision", RED), ("Recall", GOLD), ("F1", "#5B6472")]
        ):
            fig.add_bar(name=metric, x=names, y=[r[idx + 1] for r in CV_COMPARISON], marker_color=color)
        fig.update_layout(
            barmode="group",
            yaxis_range=[0, 1],
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Deployed here: **TF-IDF + SGD Classifier** — best mean CV F1 (0.897).")

        st.write("")
        st.markdown(
            f'Held-out test set &nbsp;<span style="color:{MUTED};">'
            f'({TEST_REPORT["Human"]["support"] + TEST_REPORT["AI"]["support"]} passages)</span>',
            unsafe_allow_html=True,
        )
        t1, t2, t3 = st.columns(3)
        t1.markdown(
            f'<div class="stat-box"><div class="v">{TEST_ACCURACY * 100:.0f}%</div>'
            f'<div class="l">Overall Accuracy</div></div>',
            unsafe_allow_html=True,
        )
        for box, cls, m in ((t2, "Human", TEST_REPORT["Human"]), (t3, "AI", TEST_REPORT["AI"])):
            box.markdown(
                f'<div class="stat-box" style="text-align:left;padding:0.6rem 0.8rem;">'
                f'<b>{cls}</b><div style="color:{MUTED};font-size:0.78rem;">'
                f'p {m["precision"]:.2f} · r {m["recall"]:.2f} · f1 {m["f1"]:.2f}</div></div>',
                unsafe_allow_html=True,
            )

with info_col:
    with st.container(border=True):
        card_header("inventory_2", "Model Information")
        st.write("")
        tfidf = model.named_steps["tfidf"]
        stat_row(
            [("SGD Classifier", "Model"), (f"{tfidf.ngram_range[0]}–{tfidf.ngram_range[1]}-gram TF-IDF", "Features")],
            cols=2,
        )
        st.write("")
        stat_row(
            [(f"{len(tfidf.vocabulary_):,}", "Vocabulary Terms"), ("3,960", "Training Rows")],
            cols=2,
        )

st.write("")

with st.container(border=True):
    card_header("dataset", "About the Dataset", accent=GOLD)
    st.write("")
    st.markdown(
        f"""
        <span style="color:{GREEN};">●</span> 1,980 Human&nbsp;&nbsp;
        <span style="color:{RED};">●</span> 1,980 AI &nbsp;&nbsp;
        <span style="color:{MUTED};">— balanced sample from
        <a href="{DATASET_URL}" target="_blank">RAID</a> (Robust AI Detection)</span>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    TOTAL_ROWS = 3960 + TEST_REPORT["Human"]["support"] + TEST_REPORT["AI"]["support"]
    stat_row(
        [
            (f"{TOTAL_ROWS:,}", "Total Passages"),
            ("3,960", "Training Rows"),
            (f"{TEST_REPORT['Human']['support'] + TEST_REPORT['AI']['support']:,}", "Test Rows"),
        ]
    )

st.markdown(
    f"""
    <div style="text-align:center;color:{MUTED};font-size:0.82rem;padding:1.4rem 0 0.3rem;">
        Built with Streamlit · <a href="{GITHUB_URL}" target="_blank" style="color:{BLUE};">
        {GITHUB_SVG} source on GitHub</a> · model runs locally, no text is stored.
    </div>
    """,
    unsafe_allow_html=True,
)
