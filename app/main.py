import numpy as np
import pandas as pd
import streamlit as st
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

st.title("SMS Spam Classifier")

input_text = st.text_area("Enter the message")

# 1. preprocess
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = word_tokenize(text)
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords.words("english")]
    text = " ".join(text)
    return text


transformed_text = preprocess_text(input_text)
# 2. vectorize
vector_input = tfidf.transform([transformed_text])

# 3. predict
result = model.predict(vector_input)[0]

# 4. display
if st.button("Predict"):
    if result == 0:
        st.header("Not spam")
    else:
        st.header("Spam")
