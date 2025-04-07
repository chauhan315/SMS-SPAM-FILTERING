from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_PATH = "Model"  

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model
