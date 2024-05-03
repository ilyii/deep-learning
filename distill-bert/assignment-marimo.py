import marimo

__generated_with = "0.4.10"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo

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
        mo,
        os,
    )


@app.cell
def __(load_dotenv):
    load_dotenv(".env")
    return


@app.cell
def __(defaultdict, os):
    def load(data_path, split):
            res = defaultdict(tuple)
            with open(os.path.join(data_path, f'{split}_text.txt'), encoding="utf-8") as f:
                text = f.readlines()
            
            with open(os.path.join(data_path, f'{split}_labels.txt'), encoding="utf-8") as f:
                labels = f.readlines()

            for i, (t, l) in enumerate(zip(text, labels)):
                res[i] = (t, int(l.strip()))
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
    DATAPATH = r"C:\Users\ihett\Workspace\deep-learning\tweeteval-emotion_recognition\data"
    train_data = load(DATAPATH,"train")
    val_data = load(DATAPATH,"val")
    test_data = load(DATAPATH,"test")

    with open(os.path.join(DATAPATH, "mapping.txt")) as f:
        mapping = {int(line.split()[0]): line.split()[1] for line in f}
    return DATAPATH, f, mapping, test_data, train_data, val_data


@app.cell
def __(AutoTokenizer, DataCollatorWithPadding):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator, tokenizer


@app.cell
def __(AutoModelForSequenceClassification, id2label, label2id):
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )
    return model,


@app.cell
def __():
    def compute_metrics():
        pass
    return compute_metrics,


@app.cell
def __(
    Trainer,
    TrainingArguments,
    compute_metrics,
    data_collator,
    model,
    tokenized_imdb,
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
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer, training_args


if __name__ == "__main__":
    app.run()
