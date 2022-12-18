import pandas as pd
from typing import Sequence
import torch
from torch.utils.data import Dataset, DataLoader
from config import selected_config as conf
from sklearn.model_selection import train_test_split





def remove_hashtag(phrase):
    splitted = phrase.lower().split()
    return " ".join([word for word in splitted if ('#' not in word) and ('@' not in word) and ('http' not in word)])


class TweetDataset(Dataset):

    def __init__(self, df):
        self.df = df
        self.tokenizer = conf.TOKENIZER  # Alternativa, hugging face solo tokenizer  # https://huggingface.co/docs/tokenizers/quicktour
        self.map_label = conf.MAP_LABEL
    def tokenize(self, phrase: str) -> Sequence[str]:
        tokenized = self.tokenizer.encode(phrase)
        n_token = len(tokenized)
        pad_tail = torch.ones(conf.MAX_N_TOKEN - n_token) * self.tokenizer.pad_token_id
        padded_tokenized = torch.cat([torch.tensor(tokenized), pad_tail], axis=0)
        return padded_tokenized.int(), n_token

    def __getitem__(self, i):
        # if i >= len(self):
        #   raise StopIteration
        i = i
        phrase = self.df.loc[i, "clean_tweet"]
        # phrase = remove_hashtag(phrase)
        text_label = self.df.loc[i, "label"]
        # Convert text_label to dummy variable, ex: 'sadness' -> [0, 1, 0, 0]
        # If it is not in the dictionary keys it returns unaltered
        number_label = self.map_label.get(text_label, text_label)

        tokenized, n_token = self.tokenize(phrase)
        return tokenized, torch.tensor(n_token), torch.tensor(number_label)

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

# sentiment140 = df = pd.read_csv(conf.PATH_DATASET + "/sentiment140/sentiment140.csv", encoding_errors='ignore', names=["target", "ids", "date", "flag", "user", "text"])
# sentiment140 = sentiment140.rename(columns={"target": "label", "text": "clean_tweet"})

# sentipolc =  pd.read_csv('dataset/sentipolc/training_set_sentipolc16.csv')
# sentipolc = sentipolc.rename(columns={"text": "clean_tweet"})

feelit_data = pd.read_csv(conf.PATH_DATASET + "/feelit/feelit.tsv", sep="\t")

train, test = train_test_split(feelit_data, test_size=0.2)
train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)
dataset_train = TweetDataset(train)
dataset_test = TweetDataset(test)
dataloader_train = TweetDataLoader(dataset_train, batch_size=conf.BATCH_SIZE, shuffle=True)
dataloader_test = TweetDataLoader(dataset_test, batch_size=conf.BATCH_SIZE, shuffle=True)

if __name__ == "__main__":
    for x in zip(dataloader_train, range(2)):
        print(x)


