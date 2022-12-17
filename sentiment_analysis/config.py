from transformers import AutoTokenizer
from pathlib import Path
import os
import torch

class Config():


    root = Path(__file__).resolve().parents[1]

    INTERVAL_TENSORBOARD = 100
    INTERVAL_SAVE_MODEL = 20_000
    INTERVAL_UPDATE_LR = 1000
    INTERVAL_TEST = 5_000


    # word embedding dimension = 96
    EMBEDDING_DIM = 96
    # Max length of a sentence in token
    SEED = 42
    MAX_N_TOKEN = 200
    BATCH_SIZE = 64
    N_ENCODER_BLOCK = 5
    EPOCHS = 100
    N_HEADS = 6
    # Number of class in labels
    OUTPUT_DIM = 4
    BPE_MODEL = "dbmdz/bert-base-italian-xxl-cased"
    DEVICE=torch.device('cpu')
    MAP_LABEL = {
        'joy': 0,
        'sadness': 1,
        'anger': 2,
        'fear': 3
        }
    LR_INIT = 1E-4
    LR_DECAY = 1E-1
    LR_STEP = 5e5

    def __init__(self):
        self.set_derivate_parameters()

    def set_derivate_parameters(config):
        """Set parameters which are derivate from other parameters"""
        config.PATH_CSV = str(config.root / 'dataset/feelit/feelit.tsv')
        config.ROOT_RUNS = str(config.root)
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