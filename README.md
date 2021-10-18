# FakeNewsMediaeval2021
Our proposition for the task FakeNews:Coronavirus for MediaEval 2021

### Results
### Results
rajjwal1/bert-tiny:
```bash
    task1:      val_f1=0.610, val_matthewscorrcoef=0.286
    task2:      val_f1=0.310, val_matthewscorrcoef=0.337
    task3:      val_f1=0.946, val_matthewscorrcoef=0.919
    multitasks: val_f1=0.901, val_matthewscorrcoef=0.866
```
google/electra-small-generator:
```bash
    task1: val_f1=0.670, val_matthewscorrcoef=nan.0
    task2: val_f1=0.000, val_matthewscorrcoef=nan.0
    task3: 
```


### Installation

* Build the Docker image and tag it

```bash
docker build . -t mediaeval2021
```

* Declare environment variables for the project path ($LOCAL_SOURCE_PATH), the dataset path ($LOCAL_DATA_PATH) and the model_name ($MODEL_NAME).
```bash
export LOCAL_SOURCE_PATH=/home/me/projects/mediaeval2021/
export LOCAL_DATA_PATH=/home/me/data/mediaeval2021/
export MODEL_NAME=prajjwal1/bert-tiny
```

* Use the docker image to run experiments.

```bash
docker run -it --rm  -v $LOCAL_SOURCE_PATH:/app -v $LOCAL_DATA_PATH:/data mediaeval2021 python train.py multitasks $MODEL_NAME /data/dev-1/
docker run -it --rm  -v $LOCAL_SOURCE_PATH:/app -v $LOCAL_DATA_PATH:/data mediaeval2021 python train.py task1 $MODEL_NAME /data/dev-1/dev-1-task-1.csv                                                                   
docker run -it --rm  -v $LOCAL_SOURCE_PATH:/app -v $LOCAL_DATA_PATH:/data mediaeval2021 python train.py task2 $MODEL_NAME /data/dev-1/dev-1-task-2.csv                                                                   
docker run -it --rm  -v $LOCAL_SOURCE_PATH:/app -v $LOCAL_DATA_PATH:/data mediaeval2021 python train.py task3 $MODEL_NAME /data/dev-1/dev-1-task-3.csv
```