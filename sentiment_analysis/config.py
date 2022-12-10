from transformers import AutoTokenizer
from pathlib import Path
import os

class Config():
    def __init__(config, **kwargs):
        config_variables = dict(

        root = "Path(__file__).resolve().parents[1]",

        PATH_CSV = "config.root / 'dataset/feelit/feelit.tsv'",
        # word embedding dimension = 96
        EMBEDDING_DIM = "96",
        # Max length of a sentence in token
        MAX_N_TOKEN = "200",
        BATCH_SIZE = "16",
        N_ENCODER_BLOCK = "5",
        BPE_MODEL = '"dbmdz/bert-base-italian-xxl-cased"',
        TOKENIZER = "AutoTokenizer.from_pretrained(config.BPE_MODEL)",
        MAX_ID_TOKEN = "len(config.TOKENIZER)",

        )

        # Set variable using dict above if no environment variables has not been defined
        for name_var, expression in config_variables.items():
            if name_var not in kwargs:
                config.__setattr__(name_var, eval(expression))
            else:
                print(f'Setting {name_var} = {kwargs[name_var]}')
                config.__setattr__(name_var, kwargs[name_var])

    def set_config(self):
        import builtins
        builtins.config = self