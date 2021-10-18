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
outpath = Path("/tmp/")

#%%


def task1_dataset(data_path, shuffle=False):
    cols = "id class text".split()
    df = pd.read_csv(data_path, sep=",", names=cols)
    df["class"] = df["class"] - 1
    if shuffle:
        df = df.sample(frac=1)
    df.to_csv(outpath / "task1-train.csv", index=False)


def task1_multilabels_dataset(data_path, shuffle=False):
    cols = "id class text".split()
    df = pd.read_csv(data_path, sep=",", names=cols)
    bin_labels = pd.get_dummies(df["class"])
    bin_labels.columns = classes
    df = pd.concat([df, bin_labels], axis=1)

    if shuffle:
        df = df.sample(frac=1)
    df.to_csv(outpath / "task1-train.csv", index=False)
    return classes


def task2_dataset(data_path, shuffle=False):
    cols = ["id"] + topics + ["text"]
    df = pd.read_csv(data_path, sep=",", names=cols)
    if shuffle:
        df = df.sample(frac=1)
    df.to_csv(outpath / "task2-train.csv", index=False)


def task3_dataset(data_path, shuffle=False):
    cols = ["id"] + topics + ["text"]
    df = pd.read_csv(data_path, sep=",", names=cols)

    enc = OneHotEncoder()
    X = enc.fit_transform(df[topics]).toarray().astype(int)
    bin_cols = [f"{t} + {c}" for c in classes for t in topics]
    bin_labels = pd.DataFrame(X, columns=bin_cols)
    df = pd.concat([df, bin_labels], axis=1)
    if shuffle:
        df = df.sample(frac=1)

    df.to_csv(outpath / "task3-train.csv", index=False)
    return bin_cols


def multitasks_dataset(data_path: Path):
    cols1 = task1_multilabels_dataset(data_path / "dev-1-task-1.csv")
    task2_dataset(data_path / "dev-1-task-2.csv")
    bin_cols = task3_dataset(data_path / "dev-1-task-3.csv")

    df1 = pd.read_csv(outpath / "task1-train.csv")
    df2 = pd.read_csv(outpath / "task2-train.csv")
    df3 = pd.read_csv(outpath / "task3-train.csv")

    df = pd.concat([df1[cols1], df2[topics], df3[bin_cols + ["text"]]], axis=1)
    df.to_csv(outpath / "multitask-train.csv", index=False)
    return cols1 + topics + bin_cols


def build_train(data_path, labels, model_name):
    multilabels = True
    if len(labels) == 1:
        labels = labels[0]
        multilabels = False

    datamodule = TextClassificationData.from_csv(
        "text",
        labels,
        train_file=data_path,
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
        multi_label=multilabels,
    )

    trainer = flash.Trainer(max_epochs=25)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze_unfreeze")
    results = trainer.test()
    print(results)


def task1(model_name, data_path):
    # task1_dataset(data_path)
    cols = task1_multilabels_dataset(data_path)
    build_train(str(outpath / "task1-train.csv"), cols, model_name)


def task2(model_name, data_path):
    task2_dataset(data_path)
    build_train(str(outpath / "task2-train.csv"), topics, model_name)


def task3(model_name, data_path):
    bin_cols = task3_dataset(data_path)
    build_train(str(outpath / "task3-train.csv"), bin_cols, model_name)


def multitasks(model_name, data_path):
    cols = multitasks_dataset(Path(data_path))
    build_train(str(outpath / "multitask-train.csv"), cols, model_name)


#%%

# model_name = "prajjwal1/bert-tiny", "google/electra-small-generator", "unitary/toxic-bert"
# data_path = "/home/tgirault/data/fake_news_datasets/medieval/dev-1/dev-1-task-2.csv"
# task3(model_name, data_path)

if __name__ == "__main__":
    import fire

    fire.Fire()
# %%
