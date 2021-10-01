#%%
from pathlib import Path
import pandas as pd
import torch
import flash

from torchmetrics import MatthewsCorrcoef, F1

from flash.text import TextClassificationData, TextClassifier
from flash.core.classification import Labels
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder


#%%
classes = [
    "Non-Conspiracy",
    "Discusses Conspiracy",
    "Promotes/Supports Conspiracy",
]

topics = [
    "Suppressed cures",
    "Behaviour and Mind Control",
    "Antivax",
    "Fake virus",
    "Intentional Pandemic",
    "Harmful Radiation/ Influence",
    "Population reduction",
    "New World Order",
    "Satanism",
]
#%%

def task1(model_name, data_path):
    cols = "id class text".split()
    df = pd.read_csv(data_path, sep=",", names=cols)
    df["class"] = df["class"] - 1
    df = df.sample(frac=1)

    # l = len(df)
    # train_split, eval_split = int(l * 0.8), int(l * 0.9)

    outpath = Path("/tmp/corpus/")
    df.to_csv(outpath / "train.csv", index=False)

    # df.iloc[:train_split].to_csv(outpath / "train.csv", index=False)
    # df.iloc[train_split:eval_split].to_csv(outpath / "valid.csv", index=False)
    # df.iloc[eval_split:].to_csv(outpath / "test.csv", index=False)

    datamodule = TextClassificationData.from_csv(
        "text",
        "class",
        train_file=str(outpath / "train.csv"),
        # val_file=str(outpath / "valid.csv"),
        val_split=0.2,
        backbone=model_name,
        batch_size=64,
    )

    model = TextClassifier(
        num_classes=datamodule.num_classes,
        backbone=model_name,
        metrics=[F1(datamodule.num_classes), MatthewsCorrcoef(datamodule.num_classes)],
        optimizer=torch.optim.AdamW,
        serializer=Labels(multi_label=True),
    )

    trainer = flash.Trainer(max_epochs=25)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze_unfreeze")
    trainer.test()
    # trainer.save_checkpoint(f"{model_name}-{dataset_name}.pt")
    # return model


def task2(model_name, data_path):
    outpath = Path("/tmp/corpus/")
    cols = ["id"] + topics + ["text"]
    df = pd.read_csv(data_path, sep=",", names=cols)
    df = df.sample(frac=1)
    df.to_csv(outpath / "task2-train.csv", index=False)

    datamodule = TextClassificationData.from_csv(
        "text",
        topics,
        train_file=str(outpath / "task2-train.csv"),
        val_split=0.2,
        backbone=model_name,
        batch_size=64,
    )

    # 2. Build the task
    model = TextClassifier(
        backbone=model_name,
        num_classes=datamodule.num_classes,
        metrics=[F1(datamodule.num_classes), MatthewsCorrcoef(datamodule.num_classes)],
        optimizer=torch.optim.AdamW,
        multi_label=True,
    )

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=25, gpus=torch.cuda.device_count())
    trainer.finetune(model, datamodule=datamodule, strategy="freeze_unfreeze")
    trainer.test()


def task3(model_name, data_path):
    outpath = Path("/tmp/corpus/")
    cols = ["id"] + topics + ["text"]
    df = pd.read_csv(data_path, sep=",", names=cols)

    enc = OneHotEncoder()
    X = enc.fit_transform(df[topics]).toarray().astype(int)
    bin_cols = [f"{t} + {c}" for c in classes for t in topics]
    bin_labels = pd.DataFrame(X, columns=bin_cols)
    df = pd.concat([df, bin_labels], axis=1)
    df = df.sample(frac=1)

    df.to_csv(outpath / "task3-train.csv", index=False)

    datamodule = TextClassificationData.from_csv(
        "text",
        bin_cols,
        train_file=str(outpath / "task3-train.csv"),
        val_split=0.2,
        backbone=model_name,
    )

    # 2. Build the task
    model = TextClassifier(
        backbone=model_name,
        num_classes=datamodule.num_classes,
        metrics=[F1(datamodule.num_classes), MatthewsCorrcoef(datamodule.num_classes)],
        optimizer=torch.optim.AdamW,
        multi_label=True,
    )

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=25, gpus=torch.cuda.device_count())
    trainer.finetune(model, datamodule=datamodule, strategy="freeze_unfreeze")
    trainer.test()


#%%

# model_name = "prajjwal1/bert-tiny", "google/electra-small-generator", "unitary/toxic-bert"
# data_path = "/home/tgirault/data/fake_news_datasets/medieval/dev-1/dev-1-task-2.csv"
# task3(model_name, data_path)

if __name__ == "__main__":
    import fire

    fire.Fire()