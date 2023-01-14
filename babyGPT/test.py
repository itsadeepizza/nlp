from config import selected_config as conf
from config import selected_config as conf

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
    conf.LOAD_PATH = r'runs/fit/23-01-13-10H56_sparkling_week/models'
    conf.LOAD_IDX = 17200
    conf.DEVICE = torch.device("cpu")
    conf.set_derivate_parameters()

    trainer = Trainer(None, None)

    sentence = 'Il gatto è'
    candidates = trainer.test_sample(sentence, temperature=1)
    print(candidates)