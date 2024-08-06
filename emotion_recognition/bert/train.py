"""
train.py

Training and evaluation process for the model.
"""

from transformers import Trainer, TrainingArguments
from config import TRAINING_ARGS
from model import initialize_tokenizer, initialize_model
from data import load_data, load_environment_variables

import evaluate
import numpy as np

def tokenize_data(data, tokenizer):
    """
    Tokenize the input data.

    Args:
        data (List[Dict[str, int]]): List of dictionaries containing 'text' and 'label'.
        tokenizer (AutoTokenizer): The tokenizer.

    Returns:
        List[Dict[str, np.ndarray]]: Tokenized data.
    """
    texts, labels = zip(*[(d["text"], d["label"]) for d in data])
    texts = tokenizer(list(texts), truncation=True, padding=True, return_tensors="pt")
    return [{"input_ids": text, "label": label} for text, label in zip(texts["input_ids"], labels)]

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.

    Args:
        eval_pred (tuple): Tuple containing predictions and labels.

    Returns:
        Dict[str, float]: Accuracy score.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)

def main():
    # Load environment variables
    load_environment_variables(".env")

    # Load data
    train_data = load_data("..\tweeteval-emotion_recognition\data", "train")
    val_data = load_data("..\tweeteval-emotion_recognition\data", "val")

    # Load label mapping
    with open("..\tweeteval-emotion_recognition\data/mapping.txt") as f:
        id2label = {int(line.split()[0]): line.split()[1] for line in f}

    # Initialize tokenizer and model
    tokenizer = initialize_tokenizer()
    model = initialize_model(len(id2label), id2label)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Tokenize data
    tokenized_train_data = tokenize_data(train_data, tokenizer)
    tokenized_val_data = tokenize_data(val_data, tokenizer)

    # Setup training arguments and trainer
    training_args = TrainingArguments(**TRAINING_ARGS)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()
    trainer.save_model("distilbert-base-uncased-emotion")

if __name__ == "__main__":
    main()
