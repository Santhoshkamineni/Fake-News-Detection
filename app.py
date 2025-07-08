import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

st.title("📰 Fake News Detection")
st.subheader("Enter a news article below to check if it is FAKE or REAL")

@st.cache_resource
@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/dhaminikaveti/datasets/main/fake_and_real_news_dataset.csv"
    df = pd.read_csv(url)

    # Ensure labels are either FAKE or REAL
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].str.upper()

    x = df['text']
    y = df['label']

    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    x_vectorized = tfidf.fit_transform(x)

    model = PassiveAggressiveClassifier()
    model.fit(x_vectorized, y)

    return tfidf, model


# Load model and vectorizer
tfidf, model = load_model()

# User input
user_input = st.text_area("Paste the news article here:")

# Prediction
if st.button("Check"):
    input_vec = tfidf.transform([user_input])
    prediction = model.predict(input_vec)[0]

    if prediction == 'FAKE':
        st.error("⚠️ This news is likely FAKE.")
    else:
        st.success("✅ This news is likely REAL.")
