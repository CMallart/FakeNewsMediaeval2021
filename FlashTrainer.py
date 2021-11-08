#%%
import numpy as np
import pandas as pd
from torch.optim import AdamW

from pytorch_lightning import seed_everything
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrcoef
from torchmetrics.classification.f_beta import F1

seed_everything(42)

import flash
from flash.text import TextClassificationData, TextClassifier
from flash.core.classification import Labels, Probabilities

from Trainer import Trainer

#%%
#%%


class FlashTrainer(Trainer):
    max_epochs = 25

    def get_dataloader(self, df_train, df_valid, test_path=None):
        return TextClassificationData.from_data_frame(
            "text",
            self.labels,
            train_data_frame=df_train,
            val_data_frame=df_valid,
            # val_split=0.2,
            backbone=self.backbone,
            batch_size=64,
        )

    def get_dataloader_csv(self, train_path=None, valid_path=None, test_path=None):
        return TextClassificationData.from_csv(
            "text",
            self.labels,
            train_file=train_path,
            val_file=valid_path,
            # val_split=0.2,
            backbone=self.backbone,
            batch_size=64,
        )

    def fit(self, df_train, df_valid, model_outpath=None):
        if len(self.labels) == 1:
            self.labels = self.labels[0]
            self.task.multilabels = False

        datamodule = self.get_dataloader(df_train, df_valid)

        n_classes = datamodule.num_classes
        self.model = TextClassifier(
            num_classes=n_classes,
            backbone=self.backbone,
            metrics=[
                F1(n_classes),
                MatthewsCorrcoef(n_classes),
            ],
            optimizer=AdamW,
            serializer=Probabilities(multi_label=True),  # Labels(multi_label=True),
            multi_label=self.task.multilabels,
        )

        trainer = flash.Trainer(max_epochs=self.max_epochs, gpus=self.nb_gpus)
        trainer.finetune(self.model, datamodule=datamodule, strategy="freeze_unfreeze")
        if model_outpath:
            trainer.save_checkpoint(model_outpath)

    def predict(self, df_val):
        probas = np.array(self.model.predict(df_val.text))
        y_pred = (probas >= 0.5).astype(int)
        y_true = np.vstack(df_val[self.labels].values)
        return self.classification_report(y_true, y_pred)

    def load_predict(self, model_path, test_path):
        self.model = TextClassifier.load_from_checkpoint(checkpoint_path=model_path)
        df_test = pd.read_csv(test_path)
        self.predict(df_test)


#%%
if __name__ == "__main__":
    import fire

    fire.Fire(FlashTrainer)
# %%
