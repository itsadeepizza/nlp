from config import selected_config as conf
import torch


conf.N_BATCH= 2
conf.LOAD_PATH= None #"../runs/fit/22-12-18-00H14_purple_week/models"
conf.LOAD_IDX= 0 #160000
conf.EPOCHS = 10000
conf.DEVICE = torch.device("cuda")
conf.OUTPUT_DIM = 5

from loader import dataloader_train, dataloader_test

from train import Trainer

trainer = Trainer(dataloader_train, dataloader_test)
trainer.train()