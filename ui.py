import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# download stopwords
nltk.download('stopwords')

# load dataset
df = pd.read_csv("reviews.csv")

# preprocessing function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# clean text
df['clean_text'] = df['review'].apply(preprocess)

# vectorization
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=1,
    sublinear_tf=True
)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# UI
st.title("Sentiment Analysis App 😊😡")

user_input = st.text_area("Enter any text or review:")

if st.button("Predict"):
    clean_input = preprocess(user_input)
    input_vector = vectorizer.transform([clean_input])

    prediction = model.predict(input_vector)
    proba = model.predict_proba(input_vector)
    confidence = max(proba[0]) * 100

    if prediction[0] == 1:
        st.success("Positive Sentiment 😊")
    else:
        st.error("Negative Sentiment 😡")

    st.write(f"Confidence: {confidence:.2f}%")