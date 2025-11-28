import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# ------------------------------
# 1. Load CSV dataset
# ------------------------------
dataset = load_dataset("csv", data_files="dataset.csv")
dataset = dataset["train"].train_test_split(test_size=0.2)

# ------------------------------
# 2. Tokenization
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ------------------------------
# 3. Load Model
# ------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=2
)

# ------------------------------
# 4. Metrics
# ------------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels)["f1"]
    }

# ------------------------------
# 5. Training Arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",        # replaces deprecated evaluation_strategy
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10
)

# ------------------------------
# 6. Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ------------------------------
# 7. Train the model
# ------------------------------
trainer.train()

# ------------------------------
# 8. Save the model
# ------------------------------
trainer.save_model("sentiment_model")
tokenizer.save_pretrained("sentiment_model")

print("Training complete! Model saved as 'sentiment_model/'")
