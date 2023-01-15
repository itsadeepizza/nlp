from config import selected_config as conf
conf.set_derivate_parameters()
from train import Trainer
import torch
# from loader import train_dataloader, test_dataloader
from model           import Transformer
from base_trainer    import BaseTrainer

import torch
import torch.nn as nn

import torch.optim as optim
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(conf.BPE_MODEL)

if __name__ == '__main__':
    conf.LOAD_PATH = r'runs/models/23-01-15-02H12_quaint_number'
    # conf.LOAD_PATH = r'runs/models/23-01-15-09H19_old_work'
    conf.LOAD_IDX = 339000
    # conf.LOAD_IDX = 114000
    conf.DEVICE = torch.device("cpu")


    trainer = Trainer(None, None)

    sentence = 'il gatto è'
    candidates = trainer.test_sample(sentence, temperature=1.5)
    print(candidates)