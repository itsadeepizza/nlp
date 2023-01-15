import torch
from config import selected_config as conf
conf.BATCH_SIZE = 5
conf.LOAD_PATH  = None
conf.LOAD_IDX   = 0
conf.LOAD_PATH = r'runs/models/23-01-15-09H19_old_work'
conf.LOAD_IDX = 114000
conf.LOAD_PATH = r'runs/models/23-01-15-02H12_quaint_number'
conf.LOAD_IDX = 339000
conf.DEVICE     = torch.device("cuda")
conf.set_derivate_parameters()
from train import Trainer
from loader import train_dataloader, test_dataloader

if __name__ == "__main__":



    trainer = Trainer(train_dataloader, test_dataloader)

    trainer.train_epoch()