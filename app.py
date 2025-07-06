import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model('newsgroup_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Constants
MAX_LEN = 100
labels = [f"Class {i}" for i in range(20)]  # Replace with real class names if available

# UI
st.title("📰 20 Newsgroups Text Classifier")
text = st.text_area("Enter a news article:")

if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = tf.argmax(model.predict(padded), axis=1).numpy()[0]
    st.success(f"Predicted class: {labels[pred]}")
