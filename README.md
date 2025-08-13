# HGTSynergy

## Requirements
- Python=3.8
- pytorch=1.12.1
- pandas=1.3.5
- numpy=1.24.4

## Pre-train
Upon executing this command, pre-training will commence, and once complete, the pretrained model will be saved in the checkpoints directory.
```shell
python pretrain.py
```

## Fine-tune
Upon executing this command, fine-tuning commences. The model loads the pretrained model from the checkpoints directory and then trains the drug synergy prediction model.
```shell
python main.py
```
