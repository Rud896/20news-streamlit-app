import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = load_model("newsgroup_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define category labels (ensure the order matches training)
categories = [
    "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles",
    "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
    "sci.space", "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast",
    "talk.politics.misc", "talk.religion.misc"
]

# Set the same max length as used in training (adjust if needed)
MAX_SEQUENCE_LENGTH = 755

# Streamlit UI
st.title("📰 20 Newsgroups Text Classifier")
st.markdown("Enter a news article or post, and the model will classify it into one of 20 categories.")

user_input = st.text_area("Enter your news content here:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess and predict
        input_sequence = tokenizer.texts_to_sequences(user_input)
        input_padded = pad_sequences(input_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

        prediction = model.predict(input_padded)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        st.success(f"Predicted Category: **{categories[predicted_class]}**")
        st.info(f"Model Confidence: {confidence:.2f}")
