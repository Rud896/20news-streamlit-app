import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download needed NLTK data (only for first-time local runs)
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def processing(text):
    text = text.lower()
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s\s+', ' ', text)
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(stemmed)



import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
# --- Include the preprocessing function above here ---

# Load model and tokenizer
model = tf.keras.models.load_model('newsgroup_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100
labels = [f"Class {i}" for i in range(20)]  # Replace with real class names if needed

st.title("📰 20 Newsgroups Classifier")
text = st.text_area("Enter your news text below:")

if st.button("Predict"):
    preprocessed = processing(text)
    seq = tokenizer.texts_to_sequences([preprocessed])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = tf.argmax(model.predict(padded), axis=1).numpy()[0]
    st.success(f"Predicted Class: {labels[pred]}")
