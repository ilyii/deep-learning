"""
config.py

Configuration settings and constants for the project.
"""

# Constants
DATAPATH = r"..\tweeteval-emotion_recognition\data"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "my_awesome_model"
HUGGINGFACE_TOKEN_ENV_VAR = "HFTOKEN"

# TrainingArguments parameters
TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 2,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "push_to_hub": False,
}
