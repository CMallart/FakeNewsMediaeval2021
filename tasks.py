#%%
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

#%%
class Task:
    task_name = "task"
    multilabels = True
    labels = []

    @classmethod
    def get_dataset(cls, data_path: Path):
        pass

    @classmethod
    def output_prediction(cls, probas, run_id):
        pass

    @classmethod
    def read_clean_csv(cls, path, cols):
        """ensure that multiple ',' in text won't be assumed to be separators"""
        maxsplit = len(cols) - 1
        path = Path(path)
        p = path / f"{path.name}-{cls.task_name}.csv"
        with p.open() as f:
            data = [l.strip().split(",", maxsplit) for l in f]
            return pd.DataFrame(data, columns=cols)


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

    @classmethod
    def output_prediction(cls, test_id, test_proba, run_id=1):
        # save the output in the task for
        # probas = np.array(cls.model.predict(df_test.text))
        y_min = np.full((test_proba.shape[0], 1), 0.2)
        y_pred = np.hstack([y_min, test_proba]).argmax(axis=1)
        df = pd.DataFrame(list(zip(test_id, y_pred)))
        outpath = f"./experiments/ME21FND_IRISA_{cls.task_name}_{run_id}.csv"
        df.to_csv(outpath, index=False, header=False)


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
        df = cls.read_clean_csv(data_path, cols)
        for l in Task2.labels:
            df[l] = pd.to_numeric(df[l])
        return df

    @classmethod
    def output_prediction(cls, test_id, test_proba, run_id=1):
        test_id = list(test_id)

        def proba_to_class(p):
            if p < 0.20:
                return 0
            elif p == 0.20:
                return -1
            else:
                return 1

        y_pred = np.vectorize(proba_to_class)(test_proba)
        # test_proba = np.where(test_proba < 0.2, 0, test_proba)
        # test_proba = np.where(test_proba == 0.2, -1, test_proba)
        # test_proba = np.where(test_proba > 0.2, 1, test_proba)

        df = pd.DataFrame(y_pred, columns=cls.labels)
        df["id"] = test_id
        outpath = f"./experiments/ME21FND_IRISA_{cls.task_name}_{run_id}.csv"
        df[["id"] + cls.labels].to_csv(outpath, index=False, header=False)


class Task3(Task):
    task_name = "task-3"
    labels = [f"{t} + {c}" for t in Task2.labels for c in Task1.labels]

    @classmethod
    def get_dataset(cls, data_path: Path):
        cols = ["id"] + Task2.labels + ["text"]
        df = cls.read_clean_csv(data_path, cols)

        enc = OneHotEncoder()
        X = enc.fit_transform(df[Task2.labels]).toarray().astype(int)
        bin_labels = pd.DataFrame(X, columns=Task3.labels)
        return pd.concat([df, bin_labels], axis=1)

    @classmethod
    def output_prediction(cls, test_id, test_proba, run_id=1):
        test_id = list(test_id)

        y_min = np.full((test_proba.shape[0], 1), 0.2)
        splits = np.split(test_proba, len(Task2.labels), axis=1)
        y_pred = np.vstack(
            [np.hstack([y_min, probas]).argmax(axis=1) for probas in splits]
        ).T
        df = pd.DataFrame(y_pred, columns=Task2.labels)
        df["id"] = test_id
        outpath = f"./experiments/ME21FND_IRISA_{cls.task_name}_{run_id}.csv"
        df[["id"] + Task2.labels].to_csv(outpath, index=False, header=False)


class MultiTasks(Task):
    task_name = "multitasks"
    labels = Task1.labels + Task2.labels + Task3.labels

    @classmethod
    def get_dataset(cls, data_path: Path):
        df1 = Task1.get_dataset(data_path)[Task1.labels]
        df2 = Task2.get_dataset(data_path)[Task2.labels]
        df3 = Task3.get_dataset(data_path)[Task3.labels + ["text", "id"]]
        return pd.concat([df1, df2, df3], axis=1)

    @classmethod
    def output_prediction(cls, test_id, test_proba, run_id=1):
        l1, l2, l3 = len(Task1.labels), len(Task2.labels), len(Task3.labels)
        splits = np.split(test_proba, [l1, l1 + l2, l1 + l2 + l3], axis=1)
        Task1.output_prediction(test_id, splits[0], run_id)
        Task2.output_prediction(test_id, splits[1], run_id)
        Task3.output_prediction(test_id, splits[2], run_id)

# %%
