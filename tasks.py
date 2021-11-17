from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class Task:
    task_name = "task"
    multilabels = True
    labels = []

    @classmethod
    def get_dataset(cls, data_path: Path):
        pass

    @classmethod
    def output_prediction(probas, run_id):
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
    def output_prediction(test_id, test_proba, run_id=1):
        # save the output in the task for
        # probas = np.array(self.model.predict(df_test.text))
        y_pred = np.array([np.argmax(probi)+1 if max(probi) > 0.2 else 0 for probi in test_proba]) #np.argmax(probas, axis=1)
        df = pd.DataFrame(list(zip(test_id , y_pred)))
        df.to_csv("ME21FND_IRISA_00{}.csv".format(run_id), index=False, header=False)

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
    
    def output_prediction(test_id, test_proba, run_id=1):
        # save the output in the task for
        def proba_to_class(prob):
            if prob<0.20:return 0
            elif prob == 0.20: return -1
            else: return 1
        test_id = list(test_id)
        y_pred = np.array([[test_id[i]]+[proba_to_class(pi) for pi in test_proba[i]]for i in range(len(test_proba))]) #np.argmax(probas, axis=1)
        df = pd.DataFrame(y_pred)#list(zip(test_id , y_pred)))
        # df = pd.to_numeric(df)
        df.to_csv("ME21FND_IRISA_10{}.csv".format(run_id), index=False, header=False)


class Task3(Task):
    task_name = "task-3"
    labels = [f"{t} + {c}" for t in Task2.labels for c in Task1.labels ]

    @classmethod
    def get_dataset(cls, data_path: Path):
        cols = ["id"] + Task2.labels + ["text"]
        df = cls.read_clean_csv(data_path, cols)

        enc = OneHotEncoder()
        X = enc.fit_transform(df[Task2.labels]).toarray().astype(int)
        bin_labels = pd.DataFrame(X, columns=Task3.labels)
        return pd.concat([df, bin_labels], axis=1)
    
    def output_prediction(test_id, test_proba, run_id=1):
        # save the output in the task for
        def split(lst, n=3):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        def proba_to_class(prob):
            final_pred = []
            return [np.argmax(splitted_proba) + 1 if max(splitted_proba) >= 0.20 else 0 for splitted_proba in split(prob)]
    
        test_id = list(test_id)
        y_pred = np.array([ [test_id[i]]+proba_to_class(test_proba[i])  for i in range(len(test_proba))]) #np.argmax(probas, axis=1)
        df = pd.DataFrame(y_pred)#list(zip(test_id , y_pred)))
        # df = pd.to_numeric(df)
        df.to_csv("ME21FND_IRISA_20{}.csv".format(run_id), index=False, header=False)


class MultiTasks(Task):
    task_name = "multitasks"
    labels = Task1.labels + Task2.labels + Task3.labels

    @classmethod
    def get_dataset(cls, data_path: Path):
        df1 = Task1.get_dataset(data_path)[Task1.labels]
        df2 = Task2.get_dataset(data_path)[Task2.labels]
        df3 = Task3.get_dataset(data_path)[Task3.labels + ["text"]]
        return pd.concat([df1, df2, df3], axis=1)
