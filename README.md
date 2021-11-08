# FakeNewsMediaeval2021
Our proposition for the task FakeNews:Coronavirus for MediaEval 2021

### Results
rajjwal1/bert-tiny :
```bash
    task1:      val_f1=0.610, val_matthewscorrcoef=0.286
    task2:      val_f1=0.310, val_matthewscorrcoef=0.337
    task3:      val_f1=0.946, val_matthewscorrcoef=0.919
    multitasks: val_f1=0.901, val_matthewscorrcoef=0.866
```

vinai/bertweet-covid19-base-cased
```bash
    multitasks: val_f1=0.926, val_matthewscorrcoef=0.900
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
docker build -f Dockerfile.cpu -t mediaeval2021
```

* Declare environment variables for the project path, the dataset local path and the pretrained model name, the trainer type and the task name.
```bash
export PROJECT_PATH=/home/me/projects/mediaeval2021/
export LOCAL_DATA_PATH=/home/me/data/mediaeval2021/
export MODEL=prajjwal1/bert-tiny
export TRAINER_SCRIPT=FlashTrainer.py # PromptTrainer.py
export TASK=task-1 # task-2, task-3, multitasks
```

* Use the docker image to run experiments.

```bash
docker run -it --rm  -v $PROJECT_PATH:/app -v $LOCAL_DATA_PATH:/data mediaeval2021 python FlashTrainer.py --task_name=$TASK --model_name=$MODEL - train /data
```

