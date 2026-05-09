# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# Load the IMDB dataset word index
word_index = imdb.get_word_index() ## Load the word index for the IMDB dataset, which is a dictionary mapping words to their corresponding integer indices. This word index is used to encode the movie reviews into sequences of integers that can be fed into the neural network model for training and prediction.
reverse_word_index = {value: key for key, value in word_index.items()} ## Create a reverse word index, which is a dictionary that maps integer indices back to their corresponding words. This is done by iterating through the original word_index dictionary and swapping the keys and values. The reverse_word_index is useful for decoding the encoded reviews back into human-readable text, allowing us to understand the content of the reviews after they have been processed by the model.

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=100)
    return padded_review

import streamlit as st
## streamlit app
st.title('IMDB Movie Review Sentiment Analysis') # Set the title of the Streamlit app to "IMDB Movie Review Sentiment Analysis", which will be displayed at the top of the app interface. This title provides users with a clear indication of the purpose of the app, which is to analyze the sentiment of movie reviews from the IMDB dataset.
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review') # Create a text area input field in the Streamlit app where users can enter their movie reviews. The label for this text area is "Movie Review". Users can type or paste their reviews into this field, and the input will be stored in the variable user_input for further processing when the "Classify" button is clicked.

if st.button('Classify'): # Create a button labeled "Classify" in the Streamlit app. When this button is clicked, the code block inside this if statement will be executed, which includes preprocessing the user input, making a prediction using the model, and displaying the sentiment and prediction score.

    preprocessed_input=preprocess_text(user_input) # Preprocess the user input review using the preprocess_text function, which encodes the review into a sequence of integer indices based on the word index and pads it to ensure a consistent length of 500 words. The resulting preprocessed_input is a 2D array that can be fed into the model for prediction.

    ## MAke prediction
    prediction=model.predict(preprocessed_input) # Make a prediction using the model's predict() method, which takes the preprocessed input and returns a prediction. The output is a 2D array where the first dimension corresponds to the batch size (in this case, 1 since we are predicting for a single review) and the second dimension corresponds to the predicted probability of the review being positive (a value between 0 and 1).
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')