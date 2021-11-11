# %%
from pathlib import Path
import pandas as pd
import torch
from torch.nn.modules.loss import BCELoss, CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup

# from torch.optim import AdamW
from openprompt.data_utils import InputExample
from openprompt import PromptForClassification, PromptDataLoader
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt.trainer import ClassificationRunner, GenerationRunner
from openprompt.plms import load_plm
from tqdm import tqdm

from Trainer import Trainer
import numpy as np

#%%
# model_name = "distilgpt2"
# model_name = "sshleifer/tiny-distilroberta-base"


#%%
class PromptTrainer(Trainer):
    max_epochs = 15

    label_words = {
        "false": ["false", "unlinked", "unrelated"],
        "true": ["true", "related", "linked"],
    }

    def __init__(self, task_name, backbone="prajjwal1/bert-tiny"):

        super().__init__(task_name, backbone)

        self.plm, self.tokenizer, self.model_config, self.WrapperClass = load_plm(
            "bert", backbone
        )

        self.template = ManualTemplate(
            # text='Tweet:{"placeholder":"text_a"}\nClassification:{"mask"}',
            text='Tweet: {"placeholder":"text_a"}\nLabel: {"placeholder":"text_b"}\nClassification:{"mask"}',
            tokenizer=self.tokenizer,
        )

        self.verbalizer = ManualVerbalizer(
            tokenizer=self.tokenizer,
            classes=["false", "true"],
            label_words=self.label_words,
        )

        self.model = PromptForClassification(
            plm=self.plm,
            template=self.template,
            verbalizer=self.verbalizer,
            freeze_plm=False,
        )

    def binarize_dataframe(self, df):
        """converts a dataset of labelled examples such as :
         TEXT -> [LABEL_1, ..., LABEL_N]
        to a dataset of binary labelled examples such as :
         TEXT & LABEL_i -> [0,1]
        """
        res = [
            [i, r.text, label, r[label]]
            for i, r in df.iterrows()
            for label in self.labels
        ]
        cols = "doc_id text_a text_b label".split()
        bdf = pd.DataFrame(res, columns=cols)
        bdf["guid"] = bdf.index
        bdf.label = bdf.label.astype(int)
        return bdf

    def get_data_loader(self, bdf, shuffle=True):
        dataset = bdf.apply(
            lambda x: InputExample(x.guid, x.text_a, x.text_b, x.label), axis=1
        ).tolist()
        print(bdf.head(10))
        print(bdf.dtypes)
        return PromptDataLoader(
            dataset=dataset,
            template=self.template,
            tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.WrapperClass,
            decoder_max_length=3,
            batch_size=8,
            shuffle=shuffle,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="head",
        )

    def get_optimizers(self):
        # it's always good practice to set no decay to biase and LayerNorm parameters
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters1 = [
            {
                "params": [
                    p
                    for n, p in self.model.plm.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.model.plm.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # Using different optimizer for prompt parameters and model parameters
        optimizer_grouped_parameters2 = [
            {
                "params": [
                    p
                    for n, p in self.model.template.named_parameters()
                    if "raw_embedding" not in n
                ]
            }
        ]

        optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
        optimizer2 = AdamW(optimizer_grouped_parameters2, lr=1e-3)
        return optimizer1, optimizer2

    def fit(self, df_train, df_val, model_outpath=None):
        """TODO : improve training loop with Pytorch Lightning"""
        loss_func = CrossEntropyLoss()
        # torch.nn.BCELoss()

        optimizer1, optimizer2 = self.get_optimizers()
        train_dataloader = self.get_data_loader(self.binarize_dataframe(df_train))

        for epoch in tqdm(range(self.max_epochs)):
            self.model.train()
            tot_loss = 0
            for step, inputs in tqdm(enumerate(train_dataloader)):
                if self.use_cuda:
                    inputs = inputs.cuda()
                logits = self.model(inputs)
                labels = inputs["label"]
                loss = loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()
                optimizer1.step()
                optimizer1.zero_grad()
                optimizer2.step()
                optimizer2.zero_grad()
            res = self.predict(df_val)
            print(res)

        # print(tot_loss / (step + 1))

    def predict(self, df_val):
        """evaluate the prompt model on a validation dataset"""
        bdf = self.binarize_dataframe(df_val)
        valid_dataloader = self.get_data_loader(bdf, shuffle=False)
        allpreds, alllabels = [], []
        self.model.eval()
        for inputs in tqdm(valid_dataloader):
            if self.use_cuda:
                inputs = inputs.cuda()
            logits = self.model(inputs)
            labels = inputs["label"]
            alllabels += labels.cpu().tolist()
            allpreds += torch.argmax(logits, dim=-1).cpu().tolist()

        bdf["pred"] = allpreds
        bdf["true"] = alllabels

        groups = bdf.groupby("doc_id").agg(list)
        groups = groups[groups.label.apply(len) == len(self.task.labels)]
        y_true = np.stack(groups["true"])
        y_pred = np.stack(groups["pred"])
        return self.classification_report(y_true, y_pred)

    # def runner(self, train, val, test):
    #     dl_train = self.get_data_loader(self.binarize_dataframe(train))
    #     dl_val = self.get_data_loader(self.binarize_dataframe(val))
    #     dl_test = self.get_data_loader(self.binarize_dataframe(test))
    #     runner = ClassificationRunner(
    #         self.model,
    #         self.model_config,
    #         dl_train,
    #         dl_val,
    #         dl_test,
    #         loss_function="cross_entropy",
    #     )
    #     runner.run()


#%%
# pt = PromptTrainer("task-2")
# pt.train("/home/tgirault/data/fake_news_datasets/medieval/")


#%%

if __name__ == "__main__":
    import fire

    fire.Fire(PromptTrainer)
