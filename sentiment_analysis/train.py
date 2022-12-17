import numpy as np

from config import selected_config as conf
conf.set_derivate_parameters()
from loader import dataloader_train
from model import Transformer
from base_trainer import BaseTrainer
import torch
import torch.nn as nn
import torch.optim as optim
import os

class Trainer(BaseTrainer):


    def __init__(self, train_dataloader, test_dataloader, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.init_models()
        self.criterion = nn.CrossEntropyLoss()



    def init_models(self):
        from model import Transformer

        # INITIALISING MODELS
        self.transformer = Transformer()

        self.models = [self.transformer]
        for model in self.models:
            model.to(self.device)

        if conf.LOAD_PATH is not None:
            # I added "map_location=conf.DEVICE" to deal with error in google colab when gpu is
            # not available

            transformer_w = torch.load(os.path.join(conf.LOAD_PATH, f"Transformer/Transformer_{conf.LOAD_IDX}.pth"), map_location=conf.DEVICE)
            self.transformer.load_state_dict(transformer_w)
            # self.decoder.eval()

        # OPTIMIZER
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=self.lr)



    def test(self):
        self.transformer.eval()
        list_test_loss = []
        correct_prediction = 0
        for x, labels in self.test_dataloader:
            # Move to chosen device
            x = x.to(self.device)
            labels = labels.to(self.device)
            pred = self.make_prediction(x)
            loss = self.loss_calculation(x, labels, pred, eval=True).item()

            # Calculate accuracy
            label_predicted = pred.argmax(axis=1)
            correct_prediction += (label_predicted == labels).float().mean()

            list_test_loss.append(loss)
        mean_test_loss = sum(list_test_loss) / len(list_test_loss)
        self.writer.add_scalar("test_loss", mean_test_loss, self.idx)
        self.writer.add_scalar("accuracy", correct_prediction / len(list_test_loss), self.idx)



    def make_prediction(self, x):
        """Run model"""
        pred = self.transformer(x)
        return pred

    def train_epoch(self):
        for x, labels in self.train_dataloader:
            self.transformer.train()
            # Move to chosen device
            x = x.to(self.device)
            labels = labels.to(self.device)
            # predict
            self.train_sample(x, labels)
            # Perform different opeartions at regular intervals
            if self.idx % conf.INTERVAL_TENSORBOARD == 0:
                # Write results on tensorboard
                self.log_tensorboard()
            if self.idx % conf.INTERVAL_SAVE_MODEL == 0:
                # Save models as pth
                self.save_models()
            if self.idx % conf.INTERVAL_UPDATE_LR == 0:
                # UPDATE LR
                self.update_lr()
                for g in self.optimizer.param_groups:
                    g['lr'] = self.lr

            if self.idx % conf.INTERVAL_TEST == 0:
                # Test model
                self.test()

    def train(self):
        """Whole procedure for train"""
        for epoch in range(conf.EPOCHS):
            self.train_epoch()


    def train_sample(self, x, labels):
        self.idx += conf.BATCH_SIZE
        pred = self.make_prediction(x)
        loss =  self.loss_calculation(x, labels, pred).item()
        self.mean_train_loss += loss
        print(f"Step {self.idx} - Loss = {loss:.5f}")

    def update_lr(self):
        self.lr = conf.LR_INIT * (conf.LR_DECAY ** (self.idx / conf.LR_STEP))


    def loss_calculation(self, x, labels, pred, eval=False):
        # reset gradients
        self.optimizer.zero_grad()
        # calculate loss
        loss = self.criterion(pred, labels)
        if not eval:
            # backpropagation
            loss.backward()
            # update parameters
            self.optimizer.step()
        return loss


    def save_models(self):
        for model in self.models:
            self.save_model(model, model.__class__.__name__)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_epoch()

#type "tensorboard --logdir=runs" in terminal