import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer=joblib.load("tfidf_vectorizer.pkl")

st.title("Sentiment Analysis App")

st.write("Enter any sentence below:")

text = st.text_area("Enter Text")

if st.button("Predict"):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    st.subheader("Prediction")

    if prediction == "positive":
        st.success("😊 Positive Sentiment")
    elif prediction == "negative":
        st.error("😞 Negative Sentiment")
    else:
        st.info(prediction)