from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_PATH = "Model"  

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   	
    if next(model.parameters()).device.type == "meta":
        model = model.to_empty(device)
    else:
        model = model.to(device)
    

    return tokenizer, model
