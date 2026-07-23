# 💬 AI Sentiment Analyzer (Streamlit)

An interactive Machine Learning web application built with **Streamlit**, **Scikit-Learn**, and **Python** to analyze emotional sentiment in real-time text.

---

## 📌 Project Overview

This sentiment classification engine uses a **TF-IDF Vectorizer** combined with a **Logistic Regression** model trained to classify text into:
- 😞 **Negative**
- 😐 **Neutral**
- 😊 **Positive**

---

## 🛠️ Repository Structure

```
sentiment-analysis-streamlit/
│
├── app.py                  # Main Streamlit Web Application
├── sentiment_model.pkl     # Pre-trained Logistic Regression Model
├── tfidf_vectorizer.pkl    # Pre-trained TF-IDF Vectorizer
├── sentimentData.csv       # Training & Evaluation Dataset
├── TrainModel.ipynb        # Model Training & EDA Notebook
├── requirements.txt        # Python Dependencies
├── .gitignore              # Git Ignore Configuration
└── README.md               # Project Documentation
```

---

## 🚀 Quickstart & Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/jaidevxr/sentiment-analysis-streamlit.git
cd sentiment-analysis-streamlit

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit app
streamlit run app.py
```

---

## 🌐 Streamlit Cloud Deployment

This app is ready for 1-click deployment on [Streamlit Community Cloud](https://share.streamlit.io):
1. Connect your GitHub account to Streamlit Cloud.
2. Select Repository: `jaidevxr/sentiment-analysis-streamlit`
3. Branch: `main`
4. Main file path: `app.py`
5. Click **Deploy!**
