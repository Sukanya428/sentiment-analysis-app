# 📝 Text Analysis & Sentiment Prediction

A Machine Learning based web application that analyzes text and predicts whether the sentiment is **Positive 😊** or **Negative 😡** in real time.

## 🚀 Live Features

* Enter any text or review
* Predict sentiment instantly
* Confidence score display
* Clean and simple Streamlit UI

## 🛠️ Technologies Used

* Python
* Pandas
* NLTK
* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* Streamlit

## 📂 Project Workflow

1. Load dataset from CSV file
2. Preprocess text (lowercase, remove punctuation, remove stopwords)
3. Convert text into numerical vectors using TF-IDF
4. Train Machine Learning model (Logistic Regression)
5. Predict sentiment from user input
6. Show result with confidence score

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run ui.py
```

## 📊 Example Inputs

**Positive**

* Amazing service
* Excellent camera quality
* Loved this product

**Negative**

* Worst purchase ever
* Terrible customer support
* Bad quality product

## 📌 Future Improvements

* Add Neutral sentiment class
* Better UI styling
* Larger real-world dataset
* Model deployment optimization

## 👩‍💻 Author

**Sukanya Rao**
