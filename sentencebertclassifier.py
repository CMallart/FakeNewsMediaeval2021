#Steps :
#1 - fine-tune a sentence bert on our data
#2 - use it to get embedding of sentences

"""
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers import evaluation

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
   InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.CosineSimilarityLoss(model)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)


sentences1 = ['This list contains the first column', 'With your sentences', 'You want your model to evaluate on']
sentences2 = ['Sentences contains the other column', 'The evaluator matches sentences1[i] with sentences2[i]', 'Compute the cosine similarity and compares it to scores[i]']
scores = [0.3, 0.6, 0.2]

evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

# ... Your other code to load training data

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator, evaluation_steps=500)

torch.save(model, path)

#reload model
model = SentenceTransformer('./my/path/to/model/')
"""

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch
import tqdm
import sys
import os
from sklearn.model_selection import train_test_split

criterion = nn.MSELoss()
EPOCHS = 200
optm = Adam(lr = 0.001)

model_sentence_bert = SentenceTransformer('distilbert-base-nli-mean-tokens')

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("Cuda Device Available")
  print("Name of the Cuda Device: ", torch.cuda.get_device_name())
  print("GPU Computational Capablity: ", torch.cuda.get_device_capability())


class Task1Dataset(Dataset):
    def __init__(self, dataset_file, model_encoding):
        self.df = pd.read_csv(dataset_file)
        self.df.columns= ["id", "label","text"] 
        self.embeddings = model_encoding.encode(self.df.text.tolist())

    def __len__(self):
        return len(self.df.text.tolist())

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.df.label[idx]
        return embedding, label



def swish(x):
    return x * F.sigmoid(x)

class Classifier(nn.Module):

    def __init__(self, num_classes, input_embedding_dim):
        super().__init__()
        self.num_classes = num_classes
        self.input_embedding_dim = input_embedding_dim

        self.fc1 = nn.Linear(self.input_embedding_dim, 512)
        self.b1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.b2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256,128)
        self.b3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, self.num_classes)

    def forward(self,x):

        x = swish(self.fc1(x))
        x = self.b1(x)
        x = swish(self.fc2(x))
        x = self.b2(x)
        x = swish(self.fc3(x))
        x = self.b3(x)
        x = F.sigmoid(self.fc4(x))

        return x


def train_batch(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss =criterion(output,y)
    loss.backward()
    optimizer.step()
    return loss, output


def train(dataset_file, model_encoding, batchsize, epochs, task=1):

    if task ==1:
        ## Initialize the DataSet, reate train and val
        data = Task1Dataset(os.join(dataset_file, "_train"),model_encoding)
        data_valid = Task1Dataset(os.join(dataset_file, "_val"),model_encoding)
        num_classes = 3

    elif task ==2:
        ## Initialize the DataSet, reate train and val
        data = Task1Dataset(os.join(dataset_file, "_train"),model_encoding)
        data_valid = Task1Dataset(os.join(dataset_file, "_val"),model_encoding)
        num_classes = 3

    elif task ==3:
        ## Initialize the DataSet, reate train and val
        data = Task1Dataset(os.join(dataset_file, "_train"),model_encoding)
        data_valid = Task1Dataset(os.join(dataset_file, "_val"),model_encoding)
        num_classes = 3

    else :
        print("Task not valid")
        sys.exit(1)

    ## Load the Dataset
    loader_train = DataLoader(dataset = data, batch_size = batchsize, shuffle =False)
    loader_valid = DataLoader(dataset = data_valid, batch_size = batchsize, shuffle =False)

    model = Classifier(num_classes, model_encoding)

    criterion = nn.MSELoss()
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        model.train()
        for bidx, batch in tqdm(enumerate(loader_train)):
            x_train, y_train = batch[0], batch[1]
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            loss, predictions = train_batch(model,x_train,y_train, optm, criterion)
            for idx, i in enumerate(predictions):
                i  = torch.round(i)
                if i == y_train[idx]:
                    correct += 1
            acc = (correct/len(data))
            epoch_loss+=loss
        print('Epoch {} Accuracy : {}'.format(epoch+1, acc*100))
        print('Epoch {} Loss : {}'.format((epoch+1),epoch_loss))

        valid_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for bidx, batch in tqdm(enumerate(loader_valid)):
            x_train, y_train = batch[0], batch[1]
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            
            target = model(x_train)
            loss = criterion(target,y_train)
            valid_loss = loss.item() * data.size(0)

        print(f'Epoch {epoch+1} \t\t Training Loss: {epoch_loss / len(data_train)} \t\t Validation Loss: {valid_loss / len(data_valid)}')

        torch.save(model, "./trained_classifier_task_{}_epoch_{}".format((task,epoch)))
    
    return model

def eval_model(model, model_encoding, dataset_file, task=1, batchsize=16):
    data_test = Task1Dataset(os.join(dataset_file, "_test"),model_encoding)
    loader_test = DataLoader(dataset = data_test, batch_size = batchsize, shuffle =False)

    model.eval()

    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(loader_test):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

if __name__ == "__main__":
    task = sys.argv[1]
    dataset_file = sys.argv[2]

    if os.path.isfile(os.join(dataset_file, "_train"))==False:
        df = pd.read_csv(dataset_file)
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=17, shuffle=True)
        df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=17, shuffle=True)
        df_train.to_csv(os.join(dataset_file, "_train"))
        df_val.to_csv(os.join(dataset_file, "_val"))
        df_test.to_csv(os.join(dataset_file, "_test"))

    model = train(dataset_file=dataset_file, model_encoding=model_sentence_bert , batchsize=16, epochs=20, task=task)
    
