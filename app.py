import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.title(" Next Word Predictor 📝")
st.markdown(
    """
    <p style='font-size: 1.2em; color: #e0e0e0;'>
    **Discover the next word in your sentence!** This app uses a trained LSTM model to predict the most likely next word based on your input.
    </p>
    <p style='color: #b3b3b3;'>
    Enter the beginning of a sentence, and let AI help complete your thought! Perfect for creative writing, AI exploration, or just for fun.
    </p>
    """,
    unsafe_allow_html=True
)

# Input and Prediction
input_text = st.text_input("💡 Start your phrase:", "Once upon a time")
if st.button("✨ Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    if next_word:
        st.success(f'**Predicted Next Word:** {next_word}', icon="🤖")
    else:
        st.warning("Could not predict the next word. Try a different phrase.", icon="⚠️")

# Custom Styling
st.markdown(
    """
    <style>
    /* Background color */
    .stApp {
        background-color: #1E1E1E;
        font-family: 'Arial', sans-serif;
    }

    /* Title styling */
    .stTitle h1 {
        color: #FFFFFF;
        font-size: 2.5em;
    }

    /* Input label styling */
    .stTextInput label {
        font-size: 1.1em;
        color: #FFFFFF;
    }

    /* Input box styling */
    .stTextInput input {
        background-color: #333333;
        color: #FFFFFF;
        border-radius: 5px;
        border: 1px solid #666666;
    }

    /* Button styling */
    .stButton button {
        background-color: #444444;
        color: #FFFFFF;
        font-weight: bold;
        border: none;
        padding: 0.5em 1em;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: #666666;
    }

    /* Success and warning message styling */
    .stSuccess {
        color: #A5D6A7;
        font-weight: bold;
        background-color: #333333;
        padding: 0.5em;
        border-radius: 5px;
    }
    .stWarning {
        color: #EF9A9A;
        font-weight: bold;
        background-color: #333333;
        padding: 0.5em;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
