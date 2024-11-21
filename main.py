import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import DataLoader
from accelerate import Accelerator

# Use mps for apple silicon
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load dataset
dataset = load_dataset("vishnun0027/spam-detection")

# Initialize tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare dataset for PyTorch
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Split dataset
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Define DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
eval_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

# Load pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model = model.to(device)

# Load evaluation metric
metric = load_metric("accuracy")

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).cpu().numpy()
    return metric.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,  # Optimized for memory
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    dataloader_num_workers=8,  # Optimize data loading
    fp16=False,  # Mixed precision is not yet fully supported on MPS
)

# Initialize Accelerator
accelerator = Accelerator()
model, train_dataloader, eval_dataloader = accelerator.prepare(
    model, train_dataloader, eval_dataloader
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")
