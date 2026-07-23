import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model and vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/sentiment-analysis.png", width=80)
    st.title("Model Details")
    st.markdown("""
    **Architecture**: TF-IDF Vectorizer + Logistic Regression Classifier  
    **Classes**: Negative 😞, Neutral 😐, Positive 😊  
    **Dataset**: Twitter / Product Reviews Dataset  
    """)
    st.markdown("---")
    st.markdown("### Quick Examples")
    if st.button("Example 1: 😊 Positive"):
        st.session_state["user_input"] = "This product exceeded all my expectations! Highly recommended."
    if st.button("Example 2: 😞 Negative"):
        st.session_state["user_input"] = "Terrible experience, broke after two days of usage. Very disappointed."
    if st.button("Example 3: 😐 Neutral"):
        st.session_state["user_input"] = "The order arrived on Tuesday as scheduled in a standard box."

# Main Interface
st.title("💬 AI Sentiment Analyzer")
st.markdown("Enter any sentence or review below to analyze its emotional sentiment and class probabilities in real-time.")

default_text = st.session_state.get("user_input", "")
user_text = st.text_area("Enter Text for Analysis:", value=default_text, height=120, placeholder="Type your text here...")

col1, col2 = st.columns([1, 4])
with col1:
    analyze_btn = st.button("🚀 Analyze", type="primary")

if analyze_btn or default_text:
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        text_vector = vectorizer.transform([user_text])
        prediction = model.predict(text_vector)[0]
        
        # Calculate probabilities if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(text_vector)[0]
            classes = model.classes_
            prob_dict = dict(zip(classes, probs))
        else:
            prob_dict = {}

        st.markdown("---")
        st.subheader("Analysis Results")
        
        # Display primary prediction with styled callout
        pred_lower = str(prediction).lower()
        if "pos" in pred_lower:
            st.success(f"### Result: 😊 Positive Sentiment")
        elif "neg" in pred_lower:
            st.error(f"### Result: 😞 Negative Sentiment")
        else:
            st.info(f"### Result: 😐 Neutral Sentiment")
            
        # Display probabilities if available
        if prob_dict:
            st.markdown("#### Confidence Breakdown")
            for c_name, p_val in prob_dict.items():
                st.write(f"**{c_name}**: `{p_val*100:.1f}%`")
                st.progress(float(p_val))

st.markdown("---")
st.caption("Built with Streamlit, Scikit-Learn & Python | Deployed via Streamlit Community Cloud")