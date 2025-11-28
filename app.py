from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

app = Flask(__name__)

# Load model + tokenizer
model_dir = "sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

def predict_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred_id = torch.argmax(logits, dim=-1).item()
    return "Positive" if pred_id == 1 else "Negative"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form.get("text")
        # optionally get language from form, but for now ignoring
        sentiment = predict_sentiment(text)
        return render_template("index.html", result=sentiment, input_text=text)
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
