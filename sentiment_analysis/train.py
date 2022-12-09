import os
import config
from loader import dataloader
from model import Transformer

class Trainer():
    def __init__(self):
        self.transformer = Transformer()


    def train_epoch(self):
        for x, label in dataloader:
            pred = self.transformer(x)
            print(pred)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_epoch()