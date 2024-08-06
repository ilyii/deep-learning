"""
data.py

Functions for loading and preprocessing data.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

def load_data(data_path: str, split: str) -> List[Dict[str, int]]:
    """
    Load text and labels from files and combine them into a list of dictionaries.

    Args:
        data_path (str): Path to the data directory.
        split (str): Dataset split (train, val, test).

    Returns:
        List[Dict[str, int]]: List of dictionaries with 'text' and 'label'.
    """
    res = []
    with open(os.path.join(data_path, f'{split}_text.txt'), encoding="utf-8") as f:
        text = f.readlines()
    with open(os.path.join(data_path, f'{split}_labels.txt'), encoding="utf-8") as f:
        labels = f.readlines()

    res = [{"text": t.strip(), "label": int(l.strip())} for t, l in zip(text, labels)]
    return res

def load_environment_variables(env_file: str) -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_file (str): Path to the .env file.
    """
    load_dotenv(env_file)
