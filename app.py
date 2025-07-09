import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

st.title("📰 Fake News Detection")
st.subheader("Enter a news article below to check if it is FAKE or REAL")

@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/Santhoshkamineni/Fake-News-Detection/main/fake_true.csv"

    response = requests.get(url)

    if response.status_code != 200:
        st.error("❌ Failed to load dataset. HTTP Error.")
        return None, None

    data = StringIO(response.text)
    df = pd.read_csv(data)

    # Validate necessary columns
    if 'text' not in df.columns or 'label' not in df.columns:
        st.error("❌ Dataset must contain 'text' and 'label' columns.")
        return None, None

    df = df[['text', 'label']].dropna()
    df['label'] = df['label'].str.upper()  # Standardize to uppercase

    x = df['text']
    y = df['label']

    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    x_vectorized = tfidf.fit_transform(x)

    model =
