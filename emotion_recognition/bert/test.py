"""
test.py

Evaluation and testing of the trained model.
"""

import torch
from transformers import pipeline
from data import load_data
from model import initialize_tokenizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, tokenizer, test_data, id2label):
    """
    Evaluate the model on the test data.

    Args:
        model (AutoModelForSequenceClassification): The trained model.
        tokenizer (AutoTokenizer): The tokenizer.
        test_data (List[Dict[str, int]]): The test data.
        id2label (Dict[int, str]): Dictionary mapping label ids to label names.

    Returns:
        float: Accuracy of the model.
        List[int]: Predictions made by the model.
    """
    texts, labels = zip(*[(d["text"], d["label"]) for d in test_data])
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    
    with torch.no_grad():
        model.eval()
        model.to("cpu")
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=1).tolist()

    accuracy = evaluate.load("accuracy")
    acc = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    return acc, predictions

def plot_confusion_matrix(labels, predictions, id2label):
    """
    Plot the confusion matrix.

    Args:
        labels (List[int]): True labels.
        predictions (List[int]): Model predictions.
        id2label (Dict[int, str]): Dictionary mapping label ids to label names.
    """
    cm = confusion_matrix(labels, predictions, normalize="true")
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, xticklabels=id2label.values(), yticklabels=id2label.values())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def main():
    # Load data
    test_data = load_data("..\tweeteval-emotion_recognition\data", "test")
    with open("..\tweeteval-emotion_recognition\data/mapping.txt") as f:
        id2label = {int(line.split()[0]): line.split()[1] for line in f}

    # Initialize tokenizer and model
    tokenizer = initialize_tokenizer()
    model = torch.load("distilbert-base-uncased-emotion")  # Load the saved model

    # Evaluate model
    acc, predictions = evaluate_model(model, tokenizer, test_data, id2label)
    print(f"Accuracy: {acc:.4f}")

    # Print classification report
    labels = [d["label"] for d in test_data]
    print(classification_report(labels, predictions, target_names=id2label.values()))

    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions, id2label)

if __name__ == "__main__":
    main()
