"""
model.py

Functions for initializing and configuring the model.
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from config import MODEL_NAME

def initialize_tokenizer() -> AutoTokenizer:
    """
    Initialize and return the tokenizer.

    Returns:
        AutoTokenizer: The initialized tokenizer.
    """
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def initialize_model(num_labels: int, id2label: Dict[int, str]) -> AutoModelForSequenceClassification:
    """
    Initialize and return the model.

    Args:
        num_labels (int): Number of labels.
        id2label (Dict[int, str]): Dictionary mapping label ids to label names.

    Returns:
        AutoModelForSequenceClassification: The initialized model.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id={label: id for id, label in id2label.items()}
    )
