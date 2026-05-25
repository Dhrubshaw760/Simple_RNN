import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the imbd dataset from word index
word_idx = imdb.get_word_index()
reverse_word_idx = {value: key for key, value in word_idx.items()}

# Load the pre-trained model with Tanh activation
model = load_model('simple_rnn_imdb.h5')

# Helper functions
def decoded_review(encoded_review):
    return ' '.join([reverse_word_idx.get(key-3, '?') for key in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_idx.get(word, 2) + 3 for word in words]    
    padded_review = sequence.pad_sequences([encoded_review], padding='pre', maxlen=500)
    return padded_review

## Prediction function

def predict_sentiment(review):
    padded_review = preprocess_text(review)
    
    prediction = model.predict(padded_review)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    return sentiment, prediction[0][0]



## Streamlit app
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

# User input
user_input = st.text_area("Movie Review")

if st.button("Classify"):

    preprocess_input = preprocess_text(user_input)
    # Make prediction
    pred = model.predict(preprocess_input)
    sentiment = 'Positive' if pred[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {pred[0][0]}")
else:
    st.write('Please enter a movie review.')