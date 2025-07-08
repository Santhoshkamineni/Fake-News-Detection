import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

st.title("📰 Fake News Detection")
st.subheader("Enter a news article below to check if it is FAKE or REAL")

@st.cache_resource
def load_model():
    dataset_url = "https://raw.githubusercontent.com/GeorgeMcIntire/fake_real_news_dataset/main/fake_and_real_news_dataset.csv"
    data = pd.read_csv(dataset_url, on_bad_lines='skip', encoding='latin1')

    x = data['text']
    y = data['label']

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
