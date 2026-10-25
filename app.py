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

MODEL_PATH = "models/tracetext_char_hybrid_experiment4.joblib"
BANNER_PATH = "assets/ai_or_human_banner_900x300(1).png"
GITHUB_URL = "https://github.com/varadisthedev/TraceText"
DATASET_URL = "https://huggingface.co/datasets/liamdugan/raid"

st.set_page_config(page_title="TraceText", page_icon=":material/fingerprint:", layout="wide")

# --- model -------------------------------------------------------------
@st.cache_resource
def load_model():
    # models/tracetext_char_hybrid_experiment4.joblib is first-party, committed
    # by the project owner (not user-uploaded), so loading this pickle is trusted.
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


# Training passages were filtered to 50–500 words (mean ~203 words).
# Keep inference inputs in a comparable size range instead of scoring tiny sentences.
MIN_RELIABLE_WORDS = 50
TRAINING_MAX_WORDS = 500
TARGET_CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40


def chunk_text(text: str) -> list[str]:
    """Keep normal passages intact; split only long inputs into overlapping windows.

    Windows are similar in size to the passages used during training. This reduces
    the train/inference mismatch caused by scoring 30–60-word sentence groups.
    """
    clean_text = text.strip()
    words = clean_text.split()
    if not words:
        return []
    if len(words) <= TRAINING_MAX_WORDS:
        return [clean_text]

    chunks = []
    start = 0
    step = TARGET_CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    while start < len(words):
        end = min(start + TARGET_CHUNK_WORDS, len(words))
        # Avoid leaving a final fragment shorter than the training minimum.
        if 0 < len(words) - end < MIN_RELIABLE_WORDS:
            end = len(words)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def analyze(text: str):
    word_count = len(text.split())
    if word_count < MIN_RELIABLE_WORDS:
        raise ValueError(
            f"Please enter at least {MIN_RELIABLE_WORDS} words. "
            "The current model was not trained or validated on shorter passages."
        )
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Please enter some readable text.")
    probs = model.predict_proba(chunks)[:, AI_CLASS_INDEX]  # model-estimated AI-like score per window
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


def _pipeline_parts():
    """Return the Experiment 3 word vectorizer, char vectorizer and classifier."""
    features = model.named_steps.get("features")
    classifier = model.named_steps.get("classifier")
    if features is None or classifier is None or not hasattr(classifier, "coef_"):
        return None, None, None
    transformers = dict(features.transformer_list)
    return transformers.get("word_tfidf"), transformers.get("char_tfidf"), classifier


def _ai_oriented_coefficients(classifier):
    coef = np.asarray(classifier.coef_).reshape(-1).astype(float)
    # For binary sklearn classifiers, coef_ points toward classes_[1].
    # Flip when class 1 (AI) is classes_[0] so positive always means AI.
    if list(model.classes_).index(1) == 0:
        coef = -coef
    return coef


def explain_chunk(chunk: str, top_n: int = 6):
    """Explain a chunk using active word and character n-gram contributions."""
    word_vec, char_vec, classifier = _pipeline_parts()
    if word_vec is None or char_vec is None or classifier is None:
        return [], []

    coefficients = _ai_oriented_coefficients(classifier)
    word_count = len(word_vec.vocabulary_)
    word_matrix = word_vec.transform([chunk]).tocsr()
    char_matrix = char_vec.transform([chunk]).tocsr()
    items = []
    if word_matrix.nnz:
        names = word_vec.get_feature_names_out()
        for idx, value in zip(word_matrix.indices, word_matrix.data):
            items.append((str(names[idx]), float(value * coefficients[idx])))
    if char_matrix.nnz:
        names = char_vec.get_feature_names_out()
        for idx, value in zip(char_matrix.indices, char_matrix.data):
            items.append((f"[char: {names[idx]}]", float(value * coefficients[word_count + idx])))
    ai_push = sorted((x for x in items if x[1] > 0), key=lambda x: x[1], reverse=True)[:top_n]
    human_push = sorted((x for x in items if x[1] < 0), key=lambda x: x[1])[:top_n]
    return ai_push, human_push


def highlight_chunk(chunk: str) -> str:
    """Render the entire passage green; only meaningful AI-leaning tokens turn red.

    Scores combine word/phrase and character n-gram contributions. Human-leaning,
    neutral, and otherwise unscored text stays green, so there are no uncolored gaps.
    """
    word_vec, char_vec, classifier = _pipeline_parts()
    if word_vec is None or char_vec is None or classifier is None:
        return f'<span class="human-text">{html.escape(chunk)}</span>'

    coefficients = _ai_oriented_coefficients(classifier)
    word_count = len(word_vec.vocabulary_)
    token_matches = list(re.finditer(r"\S+", chunk))
    if not token_matches:
        return f'<span class="human-text">{html.escape(chunk)}</span>'
    token_scores = np.zeros(len(token_matches), dtype=float)
    char_scores = np.zeros(len(chunk), dtype=float)

    # Word/phrase n-gram contributions are assigned to each token they cover.
    wm = word_vec.transform([chunk]).tocsr()
    word_names = word_vec.get_feature_names_out()
    for idx, value in zip(wm.indices, wm.data):
        contribution = float(value * coefficients[idx])
        feature = str(word_names[idx]).strip()
        if not feature:
            continue
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in feature.split()) + r"(?!\w)"
        for match in re.finditer(pattern, chunk, flags=re.IGNORECASE):
            for ti, token in enumerate(token_matches):
                if token.start() < match.end() and token.end() > match.start():
                    token_scores[ti] += contribution / max(1, len(feature.split()))

    # Character n-gram contributions are spread over matching character spans,
    # then averaged into the tokens they overlap. This is an approximate local
    # explanation of a linear model, not a causal attribution.
    cm = char_vec.transform([chunk]).tocsr()
    char_names = char_vec.get_feature_names_out()
    char_weights = np.zeros(len(chunk), dtype=float)
    char_hits = np.zeros(len(chunk), dtype=float)
    lower_chunk = chunk.lower()
    for idx, value in zip(cm.indices, cm.data):
        contribution = float(value * coefficients[word_count + idx])
        feature = str(char_names[idx]).lower()
        if not feature:
            continue
        pos = 0
        while True:
            found = lower_chunk.find(feature, pos)
            if found < 0:
                break
            stop = found + len(feature)
            char_weights[found:stop] += contribution
            char_hits[found:stop] += 1
            pos = found + 1
    char_scores = np.divide(char_weights, char_hits, out=np.zeros_like(char_weights), where=char_hits > 0)
    for ti, token in enumerate(token_matches):
        span = char_scores[token.start():token.end()]
        if span.size:
            token_scores[ti] += float(np.mean(span))

    # Only clearly positive tokens become red. Everything else remains green.
    positive = token_scores[token_scores > 0]
    if not positive.size:
        red_threshold = float("inf")
    else:
        red_threshold = max(float(np.quantile(positive, 0.55)) * 0.65, float(positive.max()) * 0.10, 1e-7)

    pieces = []
    cursor = 0
    for i, token in enumerate(token_matches):
        if token.start() > cursor:
            pieces.append(f'<span class="human-text">{html.escape(chunk[cursor:token.start()])}</span>')
        token_text = html.escape(token.group())
        if token_scores[i] >= red_threshold:
            pieces.append(f'<span class="ai-word" title="AI-leaning model feature">{token_text}</span>')
        else:
            pieces.append(f'<span class="human-text">{token_text}</span>')
        cursor = token.end()
    if cursor < len(chunk):
        pieces.append(f'<span class="human-text">{html.escape(chunk[cursor:])}</span>')
    return "".join(pieces)

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
    "remains essential to fully realize these benefits. These patterns deserve careful evaluation."
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
    .human-text {{ background: #E8F5EA; color: #14532D; border-radius: 2px; }}
    .ai-word {{ background: #FDECEC; color: #991B1B; border-radius: 3px; font-weight: 650; }}
    .chunk-span {{ background: #E8F5EA; border-left-color: {GREEN}; color: #14532D; }}
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
            if not text.strip():
                st.session_state.result = None
                st.warning("Enter some text first.")
            elif len(text.split()) < MIN_RELIABLE_WORDS:
                st.session_state.result = None
                st.warning(
                    f"This model was trained on passages of at least {MIN_RELIABLE_WORDS} words. "
                    f"Your text has {len(text.split())} words, so a prediction would be unreliable."
                )
            else:
                st.session_state.result = analyze(text)

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
            verdict_label = "AI-like signals" if verdict_ai else "Human-like signals"
            verdict_icon = "smart_toy" if verdict_ai else "person"
            top_score = max(ai_pct, human_pct)

            st.write("")
            c1, c2 = st.columns([1, 1], vertical_alignment="center")
            c1.markdown(
                f'<span class="verdict-pill" style="background:{verdict_color};">'
                f'<span class="tt-icon">{verdict_icon}</span> {verdict_label}</span>',
                unsafe_allow_html=True,
            )
            c2.markdown(
                f'<div style="text-align:right;"><div class="confidence-value">{top_score:.1f}%</div>'
                f'<div class="confidence-label">highest model score · not calibrated</div></div>',
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
                        dict(text=f"{ai_pct:.0f}%<br>AI-like", x=0.5, y=0.5, font_size=17, showarrow=False)
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
                f"The model score leans **{ai_pct:.1f}% AI-like / {human_pct:.1f}% human-like**, "
                "averaged across the passage windows and weighted by word count. "
                "These are model scores, not calibrated real-world probabilities or proof of authorship. "
                "The deployed classifier is word + character TF-IDF with SGDClassifier."
            )

# --- chunk breakdown -----------------------------------------------------
if st.session_state.result:
    result = st.session_state.result
    st.write("")
    with st.container(border=True):
        card_header(
            "grid_view", "Chunk-by-Chunk Breakdown",
            "Short passages (50–500 words) are scored as one passage; longer inputs use overlapping word windows.",
        )
        st.write("")
        rows = []
        for i, (chunk, prob) in enumerate(zip(result["chunks"], result["probs"]), start=1):
            # Keep the whole chunk green; highlight only AI-leaning words/phrases red.
            tag = f"AI-like score {prob * 100:.0f}%"
            rows.append(
                f'<div class="chunk-span" style="background:#E8F5EA;border-left-color:{GREEN};">'
                f'<span class="chunk-tag" style="background:{BLUE};">chunk {i} · {tag}</span>'
                f"{highlight_chunk(chunk)}</div>"
            )
        st.markdown(f'<div class="chunk-list">{"".join(rows)}</div>', unsafe_allow_html=True)
        st.caption("Green is the default for all text (including neutral/unscored text); red marks words the model's learned features lean toward AI. These highlights are approximate explanations, not proof of authorship.")

    with st.container(border=True):
        card_header(
            "manage_search", "Why the Model Scored It This Way",
            "Local feature contributions from Experiment 3 word + character TF-IDF and SGD. Positive contributions push toward AI; negative contributions push toward human.",
            accent=BLUE,
        )
        st.caption(
            "These are patterns the model learned from its training data, not proof that a phrase was written by AI. "
            "A feature's contribution depends on its value in this specific chunk."
        )
        for i, chunk in enumerate(result["chunks"], start=1):
            with st.expander(f"Chunk {i} — inspect influential words and phrases", expanded=(i == 1)):
                ai_push, human_push = explain_chunk(chunk)
                if not ai_push and not human_push:
                    st.info("No vocabulary features from the fitted TF-IDF model were active for this chunk, or the saved model does not expose linear coefficients.")
                    continue
                left, right = st.columns(2, gap="medium")
                with left:
                    st.markdown(f"**Pushes toward AI** <span style='color:{RED};'>●</span>", unsafe_allow_html=True)
                    if ai_push:
                        for feature, contribution in ai_push:
                            st.markdown(
                                f"<div style='padding:.4rem .55rem;margin:.25rem 0;border-left:3px solid {RED};background:#FDECEC;border-radius:5px;'>"
                                f"<span style='font-weight:600;color:#7F1D1D;'>{html.escape(feature)}</span>"
                                f"<span style='float:right;color:#7F1D1D;'>+{contribution:.4f}</span></div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No positive feature contributions in this chunk.")
                with right:
                    st.markdown(f"**Pushes toward human** <span style='color:{GREEN};'>●</span>", unsafe_allow_html=True)
                    if human_push:
                        for feature, contribution in human_push:
                            st.markdown(
                                f"<div style='padding:.4rem .55rem;margin:.25rem 0;border-left:3px solid {GREEN};background:#E8F5EA;border-radius:5px;'>"
                                f"<span style='font-weight:600;color:#14532D;'>{html.escape(feature)}</span>"
                                f"<span style='float:right;color:#14532D;'>{contribution:.4f}</span></div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No negative feature contributions in this chunk.")

st.write("")

# --- model performance (Experiment 4: RAID + DACTYL modern-LLM data) ---
CV_COMPARISON = [
    ("Word + Character TF-IDF + SGD (Experiment 4)", 0.896, 0.930, 0.857, 0.892),
    ("Previous RAID-only model (Experiment 3)", 0.909, 0.935, 0.878, 0.906),
]
TEST_REPORT = {
    "Human": dict(precision=0.90, recall=0.93, f1=0.91, support=792),
    "AI": dict(precision=0.92, recall=0.90, f1=0.91, support=791),
}
TEST_ACCURACY = 0.91

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
        st.caption("Deployed here: **Word + Character TF-IDF + SGD (Experiment 4)** — mean grouped-CV F1 about 0.892, trained on RAID plus modern-generator (GPT-4o, Claude, Gemini) text.")

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
        word_vec, char_vec, classifier = _pipeline_parts()
        word_vocab = len(word_vec.vocabulary_) if word_vec is not None else 0
        char_vocab = len(char_vec.vocabulary_) if char_vec is not None else 0
        stat_row(
            [("SGD Classifier", "Model"), ("Word + character TF-IDF", "Features")],
            cols=2,
        )
        st.write("")
        stat_row(
            [(f"{word_vocab + char_vocab:,}", "Combined Features"), ("7,919", "Training Rows")],
            cols=2,
        )

st.write("")

with st.container(border=True):
    card_header("dataset", "About the Dataset", accent=GOLD)
    st.write("")
    st.markdown(
        f"""
        <span style="color:{GREEN};">●</span> 3,959 Human&nbsp;&nbsp;
        <span style="color:{RED};">●</span> 3,960 AI &nbsp;&nbsp;
        <span style="color:{MUTED};">— balanced sample from
        <a href="{DATASET_URL}" target="_blank">RAID</a> and
        <a href="https://huggingface.co/datasets/ShantanuT01/DACTYL" target="_blank">DACTYL</a>
        (GPT-4o, Claude, Gemini, and other modern generators)</span>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    TOTAL_ROWS = 7919  # the 1,583 held-out rows are a subset of these 7,919 passages
    stat_row(
        [
            (f"{TOTAL_ROWS:,}", "Unique Dataset Passages"),
            (f"{TOTAL_ROWS:,}", "Final Model Fit Rows"),
            (f"{TEST_REPORT['Human']['support'] + TEST_REPORT['AI']['support']:,}", "Held-Out Test Rows (subset)"),
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
