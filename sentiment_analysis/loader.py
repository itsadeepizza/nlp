import pandas as pd
from typing import Sequence
import torch
from torch.utils.data import Dataset, DataLoader
from config import *





class TweetDataset(Dataset):

    def __init__(self, df):
        self.df = df
        self.tokenizer = tokenizer  # Alternativa, hugging face solo tokenizer  # https://huggingface.co/docs/tokenizers/quicktour

    def tokenize(self, phrase: str) -> Sequence[str]:
        tokenized = self.tokenizer.encode(phrase)
        n_token = len(tokenized)
        pad_tail = torch.ones(MAX_N_TOKEN - n_token) * self.tokenizer.pad_token_id
        padded_tokenized = torch.cat([torch.tensor(tokenized), pad_tail], axis=0)
        return padded_tokenized.int(), n_token

    def __getitem__(self, i):
        # if i >= len(self):
        #   raise StopIteration
        phrase = self.df.loc[i, "clean_tweet"]
        text_label = self.df.loc[i, "label"]
        # Convert text_label to dummy variable, ex: 'sadness' -> [0, 1, 0, 0]
        map_label = {
            'joy': 0,
            'sadness': 1,
            'anger': 2,
            'fear': 3
            }
        tokenized, n_token = self.tokenize(phrase)
        return tokenized, torch.tensor(n_token), torch.tensor(map_label[text_label])

    def __len__(self):
        return len(self.df)


class TweetDataLoader():
    """Remove excessive padding to batched samples """
    def __init__(self, *args, **kwargs):
        self.dataloader = DataLoader(*args, **kwargs)


    def __iter__(self):
        self.iter_dataloader = iter(self.dataloader)
        return self

    def __next__(self):
        batch, n_token, label = next(self.iter_dataloader)
        max_length = n_token.max()
        cropped_batch = batch[:, 0:max_length]
        return cropped_batch, label


feelit_data = pd.read_csv(PATH_CSV, sep="\t")
dataset = TweetDataset(feelit_data)
dataloader = TweetDataLoader(dataset, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    for x in dataloader:
        print(x)
