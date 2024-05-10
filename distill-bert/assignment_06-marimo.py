import marimo

__generated_with = "0.4.10"
app = marimo.App()


@app.cell
def __():
    import huggingface_hub
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
    import evaluate

    import os
    from dotenv import load_dotenv
    from collections import defaultdict
    return (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        defaultdict,
        evaluate,
        huggingface_hub,
        load_dataset,
        load_dotenv,
        os,
    )


@app.cell
def __(load_dotenv):
    load_dotenv(".env")
    return


@app.cell
def __(os):
    def load(data_path, split):
            res = list()
            with open(os.path.join(data_path, f'{split}_text.txt'), encoding="utf-8") as f:
                text = f.readlines()
            
            with open(os.path.join(data_path, f'{split}_labels.txt'), encoding="utf-8") as f:
                labels = f.readlines()

            for i, (t, l) in enumerate(zip(text, labels)):
                res.append({
                    "text": t,
                    "label": int(l.strip())
                })
            return res
    return load,


@app.cell
def __(huggingface_hub, os):
    huggingface_hub.login(os.environ["HFTOKEN"])
    return


@app.cell
def __(load, os):
    # dataset = load_dataset()
    # "label", "text"
    DATAPATH = r"..\tweeteval-emotion_recognition\data"
    train_data = load(DATAPATH,"train")
    val_data = load(DATAPATH,"val")
    test_data = load(DATAPATH,"test")

    with open(os.path.join(DATAPATH, "mapping.txt")) as f:
        id2label = {int(line.split()[0]): line.split()[1] for line in f}
    return DATAPATH, f, id2label, test_data, train_data, val_data


@app.cell
def __(AutoTokenizer, DataCollatorWithPadding):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator, tokenizer


@app.cell
def __(AutoModelForSequenceClassification, id2label):
    # 66_956_548 parameters
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(id2label), id2label=id2label, label2id={label: id for id, label in id2label.items()}
    )

    # Parameters
    model.num_parameters()
    return model,


@app.cell
def __(evaluate, tokenizer):
    import numpy as np

    accuracy = evaluate.load("accuracy")

    def tokenize(data):
        texts, labels = zip(*[(d["text"], d["label"]) for d in data])
        texts = tokenizer(list(texts), truncation=True, padding=True)
        return [{"input_ids": text, "label": label} for text, label in zip(texts["input_ids"], labels)]

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    return accuracy, compute_metrics, np, tokenize


@app.cell
def __(test_data, tokenize, train_data, val_data):
    tokenized_train_data = tokenize(train_data)
    tokenized_val_data = tokenize(val_data)
    tokenized_test_data = tokenize(test_data)
    return tokenized_test_data, tokenized_train_data, tokenized_val_data


@app.cell
def __(
    Trainer,
    TrainingArguments,
    compute_metrics,
    data_collator,
    model,
    tokenized_train_data,
    tokenized_val_data,
    tokenizer,
):
    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer, training_args


@app.cell
def __(trainer):
    trainer.save_model("distilbert-base-uncased-emotion")
    return


@app.cell
def __(tokenizer, trainer):
    from transformers import pipeline
    classifier = pipeline("text-classification", model=trainer.model, tokenizer=tokenizer)
    return classifier, pipeline


@app.cell
def __(accuracy, test_data, tokenizer, trainer):
    import torch
    model = trainer.model
    texts, labels = zip(*[(d["text"], d["label"]) for d in test_data])
    with torch.no_grad():
        model.eval()
        model.to("cpu")
        inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=1).tolist()

    acc = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    acc
    return acc, inputs, labels, model, outputs, predictions, texts, torch


@app.cell
def __(id2label, labels, predictions):
    # Accuracy per class
    from sklearn.metrics import classification_report
    print(classification_report(labels, predictions, target_names=id2label.values()))

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(labels, predictions, normalize="true")
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, xticklabels=id2label.values(), yticklabels=id2label.values())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


    return classification_report, cm, confusion_matrix, plt, sns


if __name__ == "__main__":
    app.run()

