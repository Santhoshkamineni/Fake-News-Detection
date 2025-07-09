import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# App Title
st.title("📰 Fake News Detection")
st.subheader("Enter a news article below to check if it is FAKE or REAL")

# Function to load model
@st.cache_resource
def load_model():
    # Link to your GitHub raw CSV file
    url = "https://raw.githubusercontent.com/Santhoshkamineni/Fake-News-Detection/main/fake_true.csv"
    response = requests.get(url)

    if response.status_code != 200:
        st.error("❌ Failed to load dataset. HTTP Error.")
        return None, None

    data = StringIO(response.text)
    df = pd.read_csv(data)

    # Ensure required columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        st.error("❌ Dataset must contain 'text' and 'label' columns.")
        return None, None

    # Clean the data
    df = df[['text', 'label']].dropna()
    df['label'] = df['label'].str.upper()

    # Features and labels
    x = df['text']
    y = df['label']

    # Vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    x_vectorized = tfidf.fit_transform(x)

    # Model training
    model = PassiveAggressiveClassifier()
    model.fit(x_vectorized, y)

    return tfidf, model

# Load model and vectorizer
tfidf, model = load_model()

# User input
user_input = st.text_area("Paste the news article here:")

# Prediction
if st.button("Check") and tfidf is not None and model is not None:
    input_vec = tfidf.transform([user_input])
    prediction = model.predict(input_vec)[0]

    if prediction == "FAKE":
        st.error("⚠️ This news is likely FAKE.")
    else:
        st.success("✅ This news is likely REAL.")
