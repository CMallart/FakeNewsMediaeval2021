#%%
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import flash

from flash.text import TextClassificationData, TextClassifier
from flash.core.classification import Labels, Probabilities
from torchmetrics import MatthewsCorrcoef, F1

from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

#%%
tmppath = Path("/tmp/")
#%%
class Task:
    task_name = "task"
    multilabels = True
    labels = []
    max_epochs = 25

    @classmethod
    def get_dataset(cls, data_path: Path):
        pass

    @classmethod
    def read_clean_csv(cls, path, cols):
        """ensure that multiple ',' in text won't be assumed to be separators"""
        maxsplit = len(cols) - 1
        path = Path(path)
        p = path / f"{path.name}-{cls.task_name}.csv"
        with p.open() as f:
            data = [l.strip().split(",", maxsplit) for l in f]
            return pd.DataFrame(data, columns=cols)  # .set_index("id")

    def build_train(self, train_path, valid_path, model_name, model_outpath=None):
        if len(self.labels) == 1:
            self.labels = self.labels[0]
            self.multilabels = False

        datamodule = TextClassificationData.from_csv(
            "text",
            self.labels,
            train_file=train_path,
            val_file=valid_path,
            # val_split=0.2,
            backbone=model_name,
            batch_size=64,
        )

        n_classes = datamodule.num_classes
        model = TextClassifier(
            num_classes=n_classes,
            backbone=model_name,
            metrics=[
                F1(n_classes),
                MatthewsCorrcoef(n_classes),
            ],
            optimizer=torch.optim.AdamW,
            serializer=Probabilities(multi_label=True),  # Labels(multi_label=True),
            multi_label=self.multilabels,
        )

        trainer = flash.Trainer(max_epochs=self.max_epochs)
        trainer.finetune(model, datamodule=datamodule, strategy="freeze_unfreeze")
        if model_outpath:
            trainer.save_checkpoint(model_outpath)
        return model

    def predict(self, model, test_path):
        model.eval()
        df = pd.read_csv(test_path)
        probas = np.array(model.predict(df.text))
        y_pred = (probas >= 0.5).astype(int)
        y_true = np.vstack(df[self.labels].values)
        self.classification_report(y_true, y_pred)

    def classification_report(self, y_true, y_pred):
        metrics = [precision_score, recall_score, f1_score, matthews_corrcoef]
        report = []
        for i, l in enumerate(self.labels):
            l_true = y_true[:, i]
            l_pred = y_pred[:, i]
            scores = [l, l_true.sum()] + [f(l_true, l_pred) for f in metrics]
            report.append(scores)
        cols = "label support precision recall f1-score mcc".split()
        df_report = pd.DataFrame(report, columns=cols)
        df_report = df_report.set_index("label")

        macro_avg = df_report.mean().values
        weighted_avg = np.average(df_report, weights=df_report["support"], axis=0)
        macro_avg[0] = weighted_avg[0] = df_report["support"].sum()
        avg = pd.DataFrame({"weighted_avg": weighted_avg, "macro_avg": macro_avg}).T
        avg.columns = cols[1:]
        df_report = pd.concat([df_report, avg]).round(2)
        df_report.to_csv(f"/tmp/report_{self.task_name}.tsv", sep="\t")

    def train_test_split(self, df):
        y = np.vstack(df[self.labels].values)
        train_index, tmp_index, val_index, test_index = (0, 0, 0, 0)

        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=2, test_size=0.2, random_state=17
        )
        for train_index, tmp_index in splitter.split(y, y):
            break

        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=2, test_size=0.5, random_state=17
        )
        for val_index, test_index in splitter.split(y[tmp_index], y[tmp_index]):
            break

        val_index, test_index = tmp_index[val_index], tmp_index[test_index]
        return df.iloc[train_index], df.iloc[val_index], df.iloc[test_index]

    def run(self, model_name, data_path="/data", model_outpath="/tmp/model.pt"):
        data_path = Path(data_path)
        df_dev = self.get_dataset(data_path / "dev")
        df_dev1 = self.get_dataset(data_path / "dev-1")
        df = pd.concat([df_dev1, df_dev])
        train, val, test = self.train_test_split(df)

        train_path = str(tmppath / f"{self.task_name}-train.csv")
        valid_path = str(tmppath / f"{self.task_name}-valid.csv")
        test_path = str(tmppath / f"{self.task_name}-test.csv")

        train.to_csv(train_path, index=False)
        val.to_csv(valid_path, index=False)
        test.to_csv(test_path, index=False)

        model = self.build_train(train_path, valid_path, model_name, model_outpath)
        self.predict(model, test_path)

    def load_predict(self, model_path, valid_path):
        model = TextClassifier.load_from_checkpoint(checkpoint_path=model_path)
        self.predict(model, valid_path)


class Task1(Task):
    task_name = "task-1"
    labels = ["Non-Conspiracy", "Discusses Conspiracy", "Promotes/Supports Conspiracy"]

    @classmethod
    def get_dataset(cls, data_path):
        cols = "id class text".split()
        df = cls.read_clean_csv(data_path, cols)

        if Task1.multilabels:
            bin_labels = pd.get_dummies(df["class"])
            bin_labels.columns = Task1.labels
            return pd.concat([df, bin_labels], axis=1)

        df["class"] = df["class"] - 1
        return df


class Task2(Task):
    task_name = "task-2"
    labels = [
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

    @classmethod
    def get_dataset(cls, data_path: Path):
        cols = ["id"] + Task2.labels + ["text"]
        return cls.read_clean_csv(data_path, cols)


class Task3(Task):
    task_name = "task-3"
    labels = [f"{t} + {c}" for c in Task1.labels for t in Task2.labels]

    @classmethod
    def get_dataset(cls, data_path: Path):
        cols = ["id"] + Task2.labels + ["text"]
        df = cls.read_clean_csv(data_path, cols)

        enc = OneHotEncoder()
        X = enc.fit_transform(df[Task2.labels]).toarray().astype(int)
        bin_labels = pd.DataFrame(X, columns=Task3.labels)
        return pd.concat([df, bin_labels], axis=1)


class MultiTasks(Task):
    task_name = "multitasks"
    labels = Task1.labels + Task2.labels + Task3.labels

    @classmethod
    def get_dataset(cls, data_path: Path):
        df1 = Task1.get_dataset(data_path)[Task1.labels]
        df2 = Task2.get_dataset(data_path)[Task2.labels]
        df3 = Task3.get_dataset(data_path)[Task3.labels + ["text"]]
        return pd.concat([df1, df2, df3], axis=1)


#%%
if __name__ == "__main__":
    import fire

    fire.Fire()
# %%
