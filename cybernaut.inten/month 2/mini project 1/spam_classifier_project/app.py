"""Streamlit web app for real-time spam detection.

Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH, OUTPUT_DIR
from src.predict import SpamDetector
from src.visualize import get_top_words

st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="wide")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #081120 0%, #111827 35%, #0f172a 100%);
            color: #f8fafc;
            font-family: 'Inter', sans-serif;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        p, li, label, div, span, .stMarkdown, .stDataFrame, .st-emotion-cache-1j7yxx {
            color: #f8fafc !important;
        }

        .stChatMessage, .stAlert p, .stAlert div, .stFileUploader label {
            color: #f8fafc !important;
        }

        .hero {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1.6rem 1.5rem;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.35);
            backdrop-filter: blur(8px);
        }

        .title {
            font-size: 2.7rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.06em;
            color: #f8fafc !important;
            margin-bottom: 0.35rem !important;
        }

        .subtitle {
            font-size: 1rem !important;
            color: #cbd5e1 !important;
        }

        .metric-card {
            background: linear-gradient(135deg, rgba(15,118,110,0.28), rgba(59,130,246,0.15));
            border: 1px solid rgba(45, 212, 191, 0.25);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            text-align: center;
            min-height: 110px;
        }

        .metric-card strong {
            display: block;
            font-size: 1.6rem;
            color: #f8fafc;
            margin-top: 0.4rem;
        }

        .metric-card span {
            color: #a5f3fc;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        div[data-testid="stTab"] {
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 12px 12px 0 0;
            color: #dbeafe !important;
            padding: 0.7rem 1rem;
            font-weight: 600;
        }

        div[data-testid="stTabList"] {
            gap: 0.6rem;
            margin-bottom: 1rem;
        }

        button[kind="primary"], button[kind="secondary"], button[kind="tertiary"] {
            color: #ffffff !important;
        }

        [role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.28), rgba(20, 184, 166, 0.22));
            border-color: rgba(96, 165, 250, 0.55);
            color: #f8fafc !important;
        }

        [data-testid="stVerticalBlock"] > div {
            background: transparent;
        }

        .stTextInput > div > div,
        .stTextArea > div > div,
        .stSelectbox > div > div {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 14px;
        }

        .stTextArea textarea,
        .stTextInput input,
        .stSelectbox div[data-baseweb="select"],
        .stSelectbox [data-testid="stBaseWidget"],
        .stSelectbox > div > div,
        .stSelectbox [role="combobox"],
        .stSelectbox input {
            background-color: #07111d !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border: 2px solid #38bdf8 !important;
            border-radius: 12px !important;
            box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.25) !important;
            caret-color: #ffffff !important;
        }

        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder,
        .stSelectbox input::placeholder,
        .stSelectbox [data-testid="stBaseWidget"]::placeholder {
            color: #dbeafe !important;
        }

        .stTextArea textarea:focus,
        .stTextInput input:focus,
        .stSelectbox div[data-baseweb="select"]:focus,
        .stSelectbox [role="combobox"]:focus {
            border-color: #22d3ee !important;
            box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.25) !important;
        }

        .stSelectbox label,
        .stTextArea label,
        .stTextInput label {
            color: #f8fafc !important;
            font-weight: 600 !important;
        }

        [role="listbox"],
        [data-baseweb="popover"] {
            background-color: #081827 !important;
            border: 1px solid #38bdf8 !important;
            color: #ffffff !important;
        }

        div[role="option"] {
            background-color: #081827 !important;
            color: #ffffff !important;
            padding: 10px 12px !important;
        }

        div[role="option"][aria-selected="true"],
        div[role="option"]:hover {
            background-color: #2563eb !important;
            color: #ffffff !important;
        }

        .stButton > button {
            border-radius: 12px;
            background: linear-gradient(135deg, #2563eb, #14b8a6);
            color: white;
            border: none;
            font-weight: 700;
            padding: 0.65rem 1.2rem;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 16px rgba(37, 99, 235, 0.35);
        }

        .stAlert {
            border-radius: 14px;
        }

        .stProgress > div > div {
            background: linear-gradient(90deg, #22c55e, #3b82f6);
        }

        .stDataFrame {
            background: rgba(15, 23, 42, 0.56);
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.16);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="title">📧 Spam Classifier</div>
        <div class="subtitle">Naive Bayes / Logistic Regression / SVM + TF-IDF — real-time email and SMS detection</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()


@st.cache_resource
def get_detector():
    return SpamDetector()


metric_cols = st.columns(3)
with metric_cols[0]:
    st.markdown('<div class="metric-card"><span>Model</span><strong>Hybrid</strong></div>', unsafe_allow_html=True)
with metric_cols[1]:
    st.markdown('<div class="metric-card"><span>Detection</span><strong>Real-time</strong></div>', unsafe_allow_html=True)
with metric_cols[2]:
    st.markdown('<div class="metric-card"><span>Inputs</span><strong>SMS / Email</strong></div>', unsafe_allow_html=True)


detector = get_detector()
tab1, tab2, tab3 = st.tabs(["Classify", "Model performance", "Spam keywords"])

with tab1:
    examples = {
        "(write your own)": "",
        "Spam example": "WINNER!! You have been selected for a $1000 prize. Call 09061701461 now to claim.",
        "Ham example": "Hey, are we still meeting for lunch tomorrow at 1?",
    }
    choice = st.selectbox("Try an example", list(examples))
    message = st.text_area("Message / email text", value=examples[choice], height=170)

    if st.button("Classify message", type="primary") and message.strip():
        result = detector.predict(message)
        conf = result["spam_confidence"]
        if result["label"] == "SPAM":
            st.error(f"🚨 SPAM  (spam probability: {conf:.1%})")
        else:
            st.success(f"✅ HAM / legitimate  (spam probability: {conf:.1%})")
        st.progress(min(max(conf, 0.0), 1.0))

with tab2:
    results_file = OUTPUT_DIR / "model_results.csv"
    if results_file.exists():
        st.dataframe(pd.read_csv(results_file).round(4), use_container_width=True)
    for img in ["06_model_comparison.png", "07_roc_curves.png"]:
        if (OUTPUT_DIR / img).exists():
            st.image(str(OUTPUT_DIR / img))

with tab3:
    st.write("Words with the strongest spam / ham weight (Logistic Regression).")
    img = OUTPUT_DIR / "08_top_keywords.png"
    if img.exists():
        st.image(str(img))
    clf = detector.model.named_steps.get("clf")
    if hasattr(clf, "coef_"):
        sw, sc, hw, hc = get_top_words(detector.model, 15)
        st.dataframe(pd.DataFrame({"spam_word": sw, "weight": sc.round(3)}))
