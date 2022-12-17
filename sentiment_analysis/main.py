from config import selected_config as conf
import torch


conf.N_BATCH= 2
conf.LOAD_PATH= None
conf.LOAD_IDX= 0
conf.DEVICE = torch.device("cuda")

from loader import dataloader_train, dataloader_test

from train import Trainer

trainer = Trainer(dataloader_train, dataloader_test)
trainer.train()