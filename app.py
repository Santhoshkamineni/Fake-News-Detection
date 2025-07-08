import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

st.title("📰 Fake News Detection")
st.subheader("Enter a news article below to check if it is FAKE or REAL")

@st.cache_resource
@st.cache_resource
def load_model():
    import requests
    from io import StringIO

    url = "https://raw.githubusercontent.com/dhaminikaveti/fake-news/main/fake_true.csv"
    response = requests.get(url)

    if response.status_code != 200:
        st.error("❌ Failed to load dataset. HTTP Error.")
        return None, None

    data = StringIO(response.text)
    df = pd.read_csv(data)

    # Make sure it has 'text' and 'label' columns
    if 'text' not in df.columns or 'label' not in df.columns:
        st.error("❌ Dataset must contain 'text' and 'label' columns.")
        return None, None

    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].str.upper()

    x = df["text"]
    y = df["label"]

    tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
    x_vectorized = tfidf.fit_transform(x)

    model = PassiveAggressiveClassifier()
    model.fit(x_vectorized, y)

    return tfidf, model


# Load the model and vectorizer
tfidf, model = load_model()

# User input box
user_input = st.text_area("Paste the news article here:")

# Predict on button click
if st.button("Check") and tfidf is not None and model is not None:
    input_vec = tfidf.transform([user_input])
    prediction = model.predict(input_vec)[0]

    if prediction == "FAKE":
        st.error("⚠️ This news is likely FAKE.")
    else:
        st.success("✅ This news is likely REAL.")
