import streamlit as st
import re
import string
import pandas as pd
from transformers import pipeline
from load_model import load_model


# loading tokenizer and model
tokenizer, model = load_model()


# classification object
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


# label mapping
label_mapping = {"LABEL_0": "ham",
                    "LABEL_1": "spam"}


# preprocessing function
def preprocess_text(text):
    text = text.strip().replace("\t", "").lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    char_count = len(text)
    word_count = len(text.split())
    special_char_count = sum(1 for char in text if char in "!@#$%^&*()")
    contains_url = 1 if re.search(r"www.\S+", text) else 0
    num_count = sum(1 for char in text if char.isdigit())

    return text, char_count, word_count, special_char_count, contains_url, num_count


st.title("📩 SMS Spam Classifier")
user_input = st.text_area("Enter an SMS to classify:")

if st.button("Predict"):
    if user_input:
        # Preprocess Input
        text, char_count, word_count, special_char_count, contains_url, num_count = preprocess_text(user_input)
        
        # Format Input for Model
        formatted_text = f"### Instruction:\nClassify the message as 'spam' or 'ham'.\n\n" \
                         f"### Message:\n{text}\n\n" \
                         f"### Features:\n" \
                         f"- Character Count: {char_count}\n" \
                         f"- Word Count: {word_count}\n" \
                         f"- Special Character Count: {special_char_count}\n" \
                         f"- Contains URL: {contains_url}\n" \
                         f"- Number Count: {num_count}\n\n" \
                         f"### Response:\n"

        # Get Model Prediction
        prediction = classifier(formatted_text)[0]['label']
        confidence = classifier(formatted_text)[0]['score']

        label = label_mapping.get(prediction)
        
        st.write(f"### 🏷 Prediction: {label} (Confidence: {confidence:.2f})")

    else:
        st.warning("⚠️ Please enter a message before predicting.")



