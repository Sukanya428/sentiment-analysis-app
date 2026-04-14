import pandas as pd

df = pd.read_csv("reviews.csv")
print(df.head())


import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

def preprocess(text):
    text = text.lower()  # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = text.split()
    
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

df['clean_text'] = df['review'].apply(preprocess)

print(df)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df['clean_text'])

print(X.toarray())

print(vectorizer.get_feature_names_out())
for i, text in enumerate(df['clean_text']):
    print(text)
    print(X.toarray()[i])
    print("------")


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# input features
X = vectorizer.fit_transform(df['clean_text'])

# output labels
y = df['label']

# split data
model = MultinomialNB()

model.fit(X, y)

# model create
model = MultinomialNB()

# training
model.fit(X,y)

# prediction
y_pred = model.predict(X)

# accuracy
print("Accuracy:", accuracy_score(y, y_pred))    

user_input = input("Enter your text: ")

clean_input = preprocess(user_input)

input_vector = vectorizer.transform([clean_input])

prediction = model.predict(input_vector)

if prediction[0] == 1:
    print("Positive Sentiment 😊")
else:
    print("Negative Sentiment 😡")