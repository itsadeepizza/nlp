from config import selected_config as conf
from tqdm import tqdm
from model           import Transformer
from base_trainer    import BaseTrainer
import torch.nn.functional as F
import torch
import torch.nn    as nn
import torch.optim as optim
import os


class Trainer(BaseTrainer):


    def __init__(self, train_dataloader, test_dataloader, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.train_dataloader = train_dataloader
        self.test_dataloader  = test_dataloader
        self.criterion        = nn.CrossEntropyLoss()
        self.init_models()
        self.init_logger()


    def init_models(self):

        # INITIALISING MODELS
        self.transformer = Transformer()

        self.models.append(self.transformer)
        for model in self.models:
            model.to(self.device)

        if conf.LOAD_PATH is not None:
            # I added "map_location=conf.DEVICE" to deal with error in google colab when gpu is
            # not available

            transformer_w = torch.load(os.path.join(conf.ROOT_RUNS + '/' +conf.LOAD_PATH, f"Transformer/Transformer_{conf.LOAD_IDX}.pth"), map_location=conf.DEVICE)

            self.transformer.load_state_dict(transformer_w)
            # self.decoder.eval()

        # OPTIMIZER
        self.optimizer = optim.AdamW(self.transformer.parameters(), lr=self.lr)


    def test(self):
        mean_test_loss = 0
        with torch.no_grad():
            self.transformer.eval()
            for x , i in zip(self.test_dataloader, range(conf.MAX_TEST_SAMPLE)):

                # Move to chosen device
                x         = x.to(self.device)
                _, loss   = self.make_prediction(x, loss=True)

                mean_test_loss += loss.item()
            mean_test_loss /= conf.MAX_TEST_SAMPLE
        self.writer.add_scalar("test_loss",        mean_test_loss, self.idx)


    def make_prediction(self, x, **kwargs):
        """Run model"""
        pred, loss = self.transformer(x, **kwargs)

        return pred, loss

    def train_epoch(self):

        for x in self.train_dataloader:
            self.optimizer.zero_grad()

            self.transformer.train()
            # Move to chosen device
            x = x.to(self.device)
            # predict
            self.train_sample(x)
            # Perform different operations at regular intervals
            if self.idx % conf.INTERVAL_TENSORBOARD < conf.BATCH_SIZE:
                # Write results on tensorboard
                self.log_tensorboard()

            if self.idx % conf.INTERVAL_SAVE_MODEL < conf.BATCH_SIZE:
                # Save models as pth
                self.save_models()

            if self.idx % conf.INTERVAL_UPDATE_LR < conf.BATCH_SIZE:
                # UPDATE LR
                self.update_lr()
                for g in self.optimizer.param_groups:
                    g['lr'] = self.lr

            if self.idx % conf.INTERVAL_TEST < conf.BATCH_SIZE:
                # Test model
                self.test()

    def train(self):
        """Whole procedure for train"""
        for epoch in range(conf.EPOCHS):

            self.train_epoch()


    def train_sample(self, x):
        self.idx += conf.BATCH_SIZE
        pred, loss = self.make_prediction(x, loss=True)
        loss.backward()
        self.optimizer.step()
        self.mean_train_loss += loss.item()
        print(f"Step {self.idx} - Loss = {loss:.5f}")

    def update_lr(self):
        self.lr = conf.LR_INIT * (conf.LR_DECAY ** (self.idx / conf.LR_STEP))

    def test_sample(self, x, samples_num=4, tokens_max_num=32, temperature=4, top_k_tokens=8):
        """Manually test the model on a sample of text"""
        samples = []
        print(x)
        with torch.no_grad():
            for idx in range(samples_num):
                prompt = x
                output = ''
                for _ in tqdm(range(tokens_max_num)):
                    logits = self.make_prediction(conf.TOKENIZER.encode(prompt, return_tensors='pt').to(conf.DEVICE))[0][0, -1, :].to(torch.float32) / temperature
                    logits = F.softmax(logits, dim=0)
                    top_k_logits_idxs = torch.fliplr(torch.unsqueeze(torch.argsort(logits), dim=0))[0][:top_k_tokens]
                    top_k_logits = torch.tensor([logits[logit_idx] for logit_idx in top_k_logits_idxs])
                    token_idx_predicted = top_k_logits_idxs[top_k_logits.multinomial(num_samples=1)[0].item()]
                    token_decoded = conf.TOKENIZER.decode(token_idx_predicted)
                    prompt += token_decoded
                    output += token_decoded
                    samples.append(output)
                    print(output)

        return samples






    def save_models(self):

        for model in self.models:
            self.save_model(model, model.__class__.__name__)




#type "tensorboard --logdir=runs" in terminal