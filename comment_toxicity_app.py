# streamlit app for classifying toxicity of a comment

# imports
import tensorflow as tf
import pandas as pd
import streamlit as st
import re
import io
import json
from PIL import Image

# image
image = Image.open("fry.jpg")

# text cleanup
def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
    
# load tokenizer
with open('tokenizer_toxic.json') as f:
    data = json.load(f)
    tokenizer_toxic = tf.keras.preprocessing.text.tokenizer_from_json(data)
    

# load model
model_toxic = tf.keras.models.load_model('model_toxic.h5')

# get predictions
def get_predictions(text_input):
    processed_text = preprocess_text(str(text_input))

    tokenized_text = tokenizer_toxic.texts_to_sequences([processed_text])

    padded_sentence = tf.keras.preprocessing.sequence.pad_sequences(tokenized_text, padding='post', maxlen=200)

    df_out = pd.DataFrame(data=model_toxic.predict(padded_sentence), columns=['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate'])
    st.write(df_out)


# streamlit stuff

st.image(image,use_column_width=True)

st.title('Comment Toxicity Classification Demo')
st.markdown('Using GloVe embeddings')



test_text = "I hope you have a great day."
user_input = st.text_area("Input text",test_text)


if st.button('Submit'):
    get_predictions(user_input )