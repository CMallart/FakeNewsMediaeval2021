from pathlib import Path
import pandas as pd
import numpy as np
import torch


from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from tasks import *
tmppath = Path("/tmp/")


class Trainer:
    max_epochs = 25

    def __init__(self, task_name, backbone="prajjwal1/bert-tiny"):

        self.nb_gpus = torch.cuda.device_count()
        self.use_cuda = self.nb_gpus > 0
        self.backbone = backbone

        self.task_name = task_name
        if task_name == "task-1":
            self.task = Task1
        elif task_name == "task-2":
            self.task = Task2
        elif task_name == "task-3":
            self.task = Task3
        elif task_name == "multitasks":
            self.task = MultiTasks
        else:
            raise ValueError(f"Unknown task {task_name}")

        self.labels = self.task.labels

    def fit(self, df_train, df_valid, model_outpath=None):
        pass

    def predict(self, df_val):
        pass

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
        return df_report

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

    def preprocess_data(self, data_path, save=True):
        data_path = Path(data_path)
        df_dev = self.task.get_dataset(data_path / "dev")
        df_dev1 = self.task.get_dataset(data_path / "dev-1")
        df = pd.concat([df_dev1, df_dev])

        train, val, test = self.train_test_split(df)
        if save:
            train_path = str(tmppath / f"{self.task_name}-train.csv")
            valid_path = str(tmppath / f"{self.task_name}-valid.csv")
            test_path = str(tmppath / f"{self.task_name}-test.csv")
            train.to_csv(train_path, index=False)
            val.to_csv(valid_path, index=False)
            test.to_csv(test_path, index=False)
        return train, val, test

    def train(self, data_path="/data", model_outpath="/tmp/model.pt"):
        df_train, df_val, df_test = self.preprocess_data(data_path)
        self.fit(df_train, df_val, model_outpath)
        report = self.predict(df_test)
        print(report)

