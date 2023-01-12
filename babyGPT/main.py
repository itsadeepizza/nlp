from config import selected_config as conf
conf.set_derivate_parameters()
from train import Trainer
import torch
from loader import train_dataloader, test_dataloader

if __name__ == "__main__":

    conf.BATCH_SIZE = 1
    conf.LOAD_PATH  = None
    conf.LOAD_IDX   = 0
    conf.DEVICE     = torch.device("cuda")

    trainer = Trainer(train_dataloader, test_dataloader)

    trainer.train_epoch()