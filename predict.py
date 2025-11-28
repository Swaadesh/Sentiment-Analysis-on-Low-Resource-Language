from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load your saved multilingual sentiment model
model_path = "sentiment_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_feedback(text):
    # Tokenize any language text (XLM-R handles multilingual)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # Predict using the trained model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

    # Convert the output
    return "Positive Feedback" if pred == 1 else "Negative Feedback"


# -------------- TEST EXAMPLES ------------------

# English
print(predict_feedback("This product is amazing!"))

# Telugu
print(predict_feedback("ఈ సినిమా చాలా బాగుంది."))   # Very good movie

# Hindi
print(predict_feedback("मुझे यह पसंद नहीं आया।"))  # I didn’t like it

# Tamil
print(predict_feedback("சூப்பரா இருக்கு"))  # It's awesome

# Arabic (example)
print(predict_feedback("هذا سيء للغاية"))  # This is very bad
