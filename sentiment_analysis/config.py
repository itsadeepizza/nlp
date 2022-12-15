from transformers import AutoTokenizer
from pathlib import Path
import os
import torch

class Config():


    root = Path(__file__).resolve().parents[0]

    # word embedding dimension = 96
    EMBEDDING_DIM = 96
    # Max length of a sentence in token
    MAX_N_TOKEN = 200
    BATCH_SIZE = 16
    N_ENCODER_BLOCK = 5
    BPE_MODEL = "dbmdz/bert-base-italian-xxl-cased"
    DEVICE=torch.device('cpu')

    def set_derivate_parameters(config):
        """Set parameters which are derivate from other parameters"""
        config.PATH_CSV = config.root / 'dataset/feelit/feelit.tsv'
        config.ROOT_RUNS = config.root
        config.TOKENIZER = AutoTokenizer.from_pretrained(config.BPE_MODEL)
        config.MAX_ID_TOKEN = len(config.TOKENIZER)

    def get(self, key, default_return_value=None):
        """Safe metod to get an attribute. If the attribute does not exist it returns
        None or a specified default value"""
        if hasattr(self, key):
            return self.__getattribute__(key)
        else:
            return default_return_value



selected_config = Config()