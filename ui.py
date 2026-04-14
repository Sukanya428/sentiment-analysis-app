import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# train model
model = MultinomialNB()
model.fit(X, y)

# UI
st.title("Sentiment Analysis App 😊😡")

user_input = st.text_area("Enter your review:")

if st.button("Predict"):
    clean_input = preprocess(user_input)
    input_vector = vectorizer.transform([clean_input])
    prediction = model.predict(input_vector)

    if prediction[0] == 1:
        st.success("Positive Sentiment 😊")
    else:
        st.error("Negative Sentiment 😡")