import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

st.title("📰 Fake News Detection")
st.subheader("Enter a news article below to check if it is FAKE or REAL")

@st.cache_resource
def load_model():
    fake_url = "https://raw.githubusercontent.com/selva86/datasets/master/Fake.csv"
    true_url = "https://raw.githubusercontent.com/selva86/datasets/master/True.csv"

    fake = pd.read_csv(fake_url, on_bad_lines='skip', encoding='latin1')
    true = pd.read_csv(true_url, on_bad_lines='skip', encoding='latin1')

    fake['label'] = 'FAKE'
    true['label'] = 'REAL'
    data = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

    x = data['text']
    y = data['label']

    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    x_vec = tfidf.fit_transform(x)

    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(x_vec, y)

    return model, tfidf

model, tfidf = load_model()

user_input = st.text_area("Paste the news article here:")

if st.button("Check"):
    input_vec = tfidf.transform([user_input])
    prediction = model.predict(input_vec)[0]
    if prediction == 'FAKE':
        st.error("⚠️ This news is likely FAKE.")
    else:
        st.success("✅ This news is likely REAL.")
