from transformers import AutoTokenizer
from pathlib import Path
import os


config_variables = dict(

root = "Path(__file__).resolve().parents[1]",

PATH_CSV = "root / 'dataset/feelit/feelit.tsv'",
# word embedding dimension = 96
EMBEDDING_DIM = "96",
# Max length of a sentence in token
MAX_N_TOKEN = "200",
BATCH_SIZE = "16",
N_ENCODER_BLOCK = "5",
BPE_MODEL = '"dbmdz/bert-base-italian-xxl-cased"',
tokenizer = "AutoTokenizer.from_pretrained(BPE_MODEL)",
MAX_ID_TOKEN = "len(tokenizer)",

)

# Set variable using dict above if no environment variables has not been defined
for name_var, expression in config_variables.items():
    if os.getenv(name_var) is None:
        exec(name_var + " = " + expression)
    else:
        exec(name_var + " = " + os.getenv(name_var))

