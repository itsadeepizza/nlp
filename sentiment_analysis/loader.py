import pandas as pd
from io import StringIO


from typing import Sequence
import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

# Path of the root directory of the project
root = Path(__file__).resolve().parents[1]
curr_dir = Path(__file__).resolve().parents[0]

# word embedding dimension = 96


def positional_encoding_word(position, d_model=96):
    dimension_index = torch.arange(d_model)
    even_idx = torch.sin(position / (10000 ** (2 * (dimension_index) / d_model)))
    odd_idx = torch.cos(position / (10000 ** (2 * (dimension_index) / d_model)))
    odd_mask = (torch.ones(d_model) - torch.pow(-1, dimension_index)) / 2
    even_mask = torch.ones(d_model) - odd_mask
    return even_idx * even_mask + odd_idx * odd_mask


def positional_encoding_sentence(length_sentence, d_model=96):
    to_stack = [positional_encoding_word(idx_word) for idx_word in range(length_sentence)]
    return torch.stack(to_stack)


def remove_hashtag(phrase):
    splitted = phrase.split()
    return " ".join([word for word in splitted if ('#' not in word) and ('@' not in word) and ('http' not in word)])


class TweetDataset(Dataset):

    def __init__(self, df):
        self.df = df
        self.nlp = spacy.load("it_core_news_sm")  # Alternativa, hugging face solo tokenizer  # https://huggingface.co/docs/tokenizers/quicktour

    def tokenize(self, phrase: str) -> Sequence[str]:
        phrase = phrase.lower()
        phrase = remove_hashtag(phrase)
        tokenized = self.nlp.tokenizer(phrase)
        processed = []
        for token in tokenized:
            if (token.is_punct or token.is_digit):
                continue
            token = self.nlp(str(token)).vector
            processed.append(token)
        embedding = torch.tensor(np.array(processed))
        len_phrase = embedding.shape[0]
        positions = positional_encoding_sentence(len_phrase)
        return embedding + positions

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
        label = torch.zeros(4)
        label[map_label[text_label]] = 1
        tokenized = self.tokenize(phrase)
        return tokenized, label

    def __len__(self):
        return len(self.df)

path_csv = root / 'dataset/feelit/feelit.tsv'
feelit_data = pd.read_csv(path_csv, sep="\t")
dataset = TweetDataset(feelit_data)

if __name__ == "__main__":
    for x in dataset:
        print(x)