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
    conf.LOAD_PATH = r'runs/models/23-01-15-18H40_orange_man'
    conf.LOAD_IDX = 849000
    conf.DEVICE = torch.device("cpu")


    trainer = Trainer(None, None)

    sentence = 'negli anni cinquanta il convento venne demolito e fu costruita una nuova'
    # sentence = '''Della sua produzione giovanile sono conducibili'''
    sentence = 'Il mio gatto è'
    candidates = trainer.test_sample(sentence, temperature=1)
    print(candidates)