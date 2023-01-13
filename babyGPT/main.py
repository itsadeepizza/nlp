import torch
from config import selected_config as conf
conf.BATCH_SIZE = 2
conf.LOAD_PATH  = None
conf.LOAD_IDX   = 0
conf.DEVICE     = torch.device("cuda")
conf.set_derivate_parameters()
from train import Trainer
from loader import train_dataloader, test_dataloader

if __name__ == "__main__":



    trainer = Trainer(train_dataloader, test_dataloader)

    trainer.train_epoch()