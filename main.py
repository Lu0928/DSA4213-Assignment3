# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# =====================================================================================
# Ensure all dependencies are installed in your environment first.
# In a Kaggle/Colab notebook, you would run:
# !pip install transformers datasets torch evaluate peft accelerate -q
# =====================================================================================

import torch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# =====================================================================================
# 1. Configuration and Initialization
# =====================================================================================

MODEL_CHECKPOINT = "distilbert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# =====================================================================================
# 2. Dataset Loading and Preprocessing
# =====================================================================================

imdb_dataset = load_dataset("imdb")
train_dataset = imdb_dataset["train"].shuffle(seed=42)
test_dataset = imdb_dataset["test"].shuffle(seed=42)

print(f"\nTrain dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print("\nDataset sample:")
print(train_dataset[0])

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =====================================================================================
# 3. Evaluation Metrics
# =====================================================================================

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

# =====================================================================================
# Experiment 0: Pretrained Model Baseline (No Fine-Tuning)
# =====================================================================================
print("\n" + "="*50)
print("Experiment 0: Pretrained Model Baseline Evaluation")
print("="*50)

no_finetuning_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=2
).to(DEVICE)

baseline_args = TrainingArguments(
    output_dir="./results/baseline_eval",
    per_device_eval_batch_size=16,
    report_to="none"
)

baseline_trainer = Trainer(
    model=no_finetuning_model,
    args=baseline_args,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nEvaluating pretrained model without fine-tuning...")
no_finetuning_eval_results = baseline_trainer.evaluate()
print(f"Pretrained Model Evaluation Results: {no_finetuning_eval_results}")

# =====================================================================================
# Experiment 1: Full Fine-Tuning
# =====================================================================================
print("\n" + "="*50)
print("Experiment 1: Starting Full Fine-Tuning")
print("="*50)

full_finetuning_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=2
).to(DEVICE)

# Define training hyperparameters for full fine-tuning
full_finetuning_args = TrainingArguments(
    output_dir="./results/full_finetuning",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,                   # <-- MODIFIED: Changed from 1 to 3
    weight_decay=0.01,
    logging_strategy="epoch",             # <-- ADDED: Log training loss at the end of each epoch
    eval_strategy="epoch",                # Evaluate after every training epoch
    save_strategy="epoch",                # Save model after every epoch
    load_best_model_at_end=True,          # Keep the best model (based on eval metrics)
    report_to="none"
)

full_finetuning_trainer = Trainer(
    model=full_finetuning_model,
    args=full_finetuning_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nStarting full fine-tuning on the complete dataset for 3 epochs...")
full_finetuning_trainer.train()

print("\nEvaluating final best full fine-tuning model...")
full_finetuning_eval_results = full_finetuning_trainer.evaluate()
print(f"Full Fine-Tuning Evaluation Results: {full_finetuning_eval_results}")

# =====================================================================================
# Experiment 2: Parameter-Efficient Fine-Tuning (LoRA)
# =====================================================================================
print("\n" + "="*50)
print("Experiment 2: Starting Fine-Tuning with LoRA")
print("="*50)

base_model_for_lora = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=2
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_lin", "k_lin", "v_lin"]
)

lora_model = get_peft_model(base_model_for_lora, lora_config).to(DEVICE)
print("\nLoRA Model Parameter Analysis:")
lora_model.print_trainable_parameters()

lora_training_args = TrainingArguments(
    output_dir="./results/lora_finetuning",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,                   # <-- MODIFIED: Changed from 1 to 3
    weight_decay=0.01,
    logging_strategy="epoch",             # <-- ADDED: Log training loss at the end of each epoch
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

lora_trainer = Trainer(
    model=lora_model,
    args=lora_training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nStarting LoRA fine-tuning on the complete dataset for 3 epochs...")
lora_trainer.train()

print("\nEvaluating final best LoRA fine-tuning model...")
lora_eval_results = lora_trainer.evaluate()
print(f"LoRA Fine-Tuning Evaluation Results: {lora_eval_results}")

# =====================================================================================
# Summary of Experiment Results
# =====================================================================================
print("\n" + "="*50)
print("Summary of Experiment Results")
print("="*50)

print(f"Pretrained (no fine-tuning) -> Accuracy: {no_finetuning_eval_results['eval_accuracy']:.4f}, F1 Score: {no_finetuning_eval_results['eval_f1']:.4f}")
print(f"Full fine-tuning (best model after 3 epochs) -> Accuracy: {full_finetuning_eval_results['eval_accuracy']:.4f}, F1 Score: {full_finetuning_eval_results['eval_f1']:.4f}")
print(f"LoRA fine-tuning (best model after 3 epochs) -> Accuracy: {lora_eval_results['eval_accuracy']:.4f}, F1 Score: {lora_eval_results['eval_f1']:.4f}")

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
print("\n" + "="*50)
print("Starting Interactive Sentiment Prediction")
print("="*50)

print("Loading the trained LoRA model for inference...")
best_lora_model_path = lora_trainer.state.best_model_checkpoint
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=2)
inference_model = PeftModel.from_pretrained(base_model, best_lora_model_path).to(DEVICE)
inference_model.eval()
print(f"Model loaded from: {best_lora_model_path}")

def predict_sentiment(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    label_map = {0: "Negative Sentiment", 1: "Positive Sentiment"}
    return label_map[prediction]

test_sentences = [
    "This movie was a complete waste of my time. The acting was terrible and the plot was boring.",
    "I absolutely loved this film! The visuals were stunning and the story was heartwarming.",
    "The book is always better than the movie.",
    "It was  bad, actually quite boring in some parts.",
    "A masterpiece of modern cinema. I was captivated from beginning to end."
]

print("\nStarting test on sample sentences:")
for sentence in test_sentences:
    prediction = predict_sentiment(sentence, inference_model, tokenizer, DEVICE)
    print(f"  - Sentence: \"{sentence}\"")
    print(f"    Prediction: {prediction}\n")