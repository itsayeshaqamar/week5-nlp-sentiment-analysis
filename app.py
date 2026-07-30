import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
from pathlib import Path

from utils import clean_text

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD CUSTOM CSS
# ==========================================================

def load_css(file_name: str):
    css_path = Path(__file__).parent / file_name
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ==========================================================
# LOAD DATA & MODEL (cached so it only loads once)
# ==========================================================

@st.cache_data
def load_data():
    return pd.read_csv("amazon_reviews.csv")

@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

df = load_data()
model, vectorizer = load_model()

# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

nav_items = [
    ("Home", "🏠  Home"),
    ("Data Overview", "📊  Data Overview"),
    ("Sentiment Predictor", "🤖  Sentiment Predictor"),
]

with st.sidebar:
    st.markdown(
        "<div class='sidebar-logo'>🛒<span>SENTIMENT AI</span></div>",
        unsafe_allow_html=True
    )

    for key, label in nav_items:
        is_active = st.session_state.page == key
        if st.button(
            label,
            key=f"nav_{key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = key
            st.rerun()

page = st.session_state.page

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "Home":

    st.markdown("""
    <div class="hero-banner">
        <h1>🛒 Amazon Review Sentiment Analysis</h1>
        <p>Turning raw customer reviews into instant sentiment insights using NLP & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    total = len(df)
    pos = int((df["feedback"] == 1).sum())
    neg = int((df["feedback"] == 0).sum())
    pos_pct = (pos / total * 100) if total else 0

    def metric_card(col, icon, label, value, color):
        col.markdown(f"""
        <div class="metric-card" style="border-top:4px solid {color};">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "📦", "Total Reviews", f"{total:,}", "#6C63FF")
    metric_card(c2, "😊", "Positive Reviews", f"{pos:,}", "#22c55e")
    metric_card(c3, "😞", "Negative Reviews", f"{neg:,}", "#ef4444")
    metric_card(c4, "📈", "Positive Rate", f"{pos_pct:.1f}%", "#f59e0b")

    st.write("")
    st.markdown("### 📋 Project Overview")
    st.markdown("""
    This project performs **Sentiment Analysis** on Amazon Alexa product reviews using
    **Natural Language Processing (NLP)** and **Machine Learning**. Every review is
    classified as either **Positive** or **Negative**, using a trained classifier built
    on **TF-IDF** text features.
    """)

    st.write("")
    st.markdown("### 🔄 Machine Learning Workflow")

    steps = [
        ("1", "🧹", "Data Cleaning", "Remove noise, punctuation & irrelevant characters"),
        ("2", "✂️", "Text Preprocessing", "Tokenize, lowercase & remove stopwords"),
        ("3", "🔢", "TF-IDF Vectorization", "Convert cleaned text into numeric features"),
        ("4", "🤖", "Model Training", "Multinomial Naive Bayes & Logistic Regression"),
        ("5", "📊", "Model Evaluation", "Accuracy, F1-score & confusion matrix"),
        ("6", "🔮", "Sentiment Prediction", "Predict the sentiment of any new review"),
    ]

    cols = st.columns(3)
    for i, (num, icon, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div class="step-icon">{icon}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================================
# DATA OVERVIEW PAGE
# ==========================================================

elif page == "Data Overview":

    st.markdown("""
    <div class="page-header">
        <h1>📊 Data Overview</h1>
        <p>Exploring the class balance and the language behind each sentiment</p>
    </div>
    """, unsafe_allow_html=True)

    sentiment_counts = df["feedback"].value_counts()
    pos_count = int(sentiment_counts.get(1, 0))
    neg_count = int(sentiment_counts.get(0, 0))

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Sentiment Class Distribution")
            fig = go.Figure(data=[go.Pie(
                labels=["Positive", "Negative"],
                values=[pos_count, neg_count],
                hole=0.55,
                marker=dict(colors=["#22c55e", "#ef4444"]),
                textinfo="label+percent"
            )])
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="#1f1147"),
                margin=dict(t=10, b=10, l=10, r=10),
                height=340
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Review Counts")
            fig2 = go.Figure(data=[go.Bar(
                x=["Positive", "Negative"],
                y=[pos_count, neg_count],
                marker_color=["#22c55e", "#ef4444"],
                text=[pos_count, neg_count],
                textposition="auto"
            )])
            fig2.update_layout(
                template="plotly_white",
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="#1f1147"),
                margin=dict(t=10, b=10, l=10, r=10),
                height=340,
                yaxis_title="Number of Reviews"
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.write("")
    st.markdown("### ☁️ Word Clouds")

    wc_col1, wc_col2 = st.columns(2)

    with wc_col1:
        with st.container(border=True):
            st.markdown("**😊 Positive Reviews**")
            image_path_pos = Path(__file__).parent / "1.png"
            if image_path_pos.exists():
                st.image(str(image_path_pos), width="stretch")
            else:
                st.warning("⚠️ Positive word cloud image '1.png' not found in the project directory.")

    with wc_col2:
        with st.container(border=True):
            st.markdown("**😞 Negative Reviews**")
            image_path_neg = Path(__file__).parent / "2.png"
            if image_path_neg.exists():
                st.image(str(image_path_neg), width="stretch")
            else:
                st.warning("⚠️ Negative word cloud image '2.png' not found in the project directory.")

# ==========================================================
# SENTIMENT PREDICTOR PAGE
# ==========================================================

else:

    st.markdown("""
    <div class="page-header">
        <h1>🤖 Sentiment Predictor</h1>
        <p>Type any review below and get an instant prediction with a confidence score</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        review = st.text_area(
            "Enter a review",
            height=160,
            placeholder="e.g. This product exceeded my expectations, works flawlessly!"
        )
        predict_clicked = st.button("🔮 Predict Sentiment", use_container_width=True)

    if predict_clicked:
        if review.strip() == "":
            st.warning("⚠️ Please enter a review before predicting.")
        else:
            cleaned = clean_text(review)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector)[0]
            confidence = probability.max() * 100

            st.write("")
            if prediction == 1:
                st.markdown("""
                <div class="result-card result-positive">
                    <div class="result-emoji">😊</div>
                    <div class="result-label">Positive Review</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card result-negative">
                    <div class="result-emoji">😞</div>
                    <div class="result-label">Negative Review</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown(f"**Confidence Score: {confidence:.2f}%**")
            st.progress(min(int(confidence), 100))

            with st.expander("🔍 View Cleaned Review Text"):
                st.code(cleaned)