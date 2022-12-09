from transformers import AutoTokenizer
from pathlib import Path

root = Path(__file__).resolve().parents[1]

PATH_CSV = root / 'dataset/feelit/feelit.tsv'
# word embedding dimension = 96
EMBEDDING_DIM = 96
# Max length of a sentence in token
MAX_N_TOKEN = 200
BATCH_SIZE = 16
N_ENCODER_BLOCK = 5
BPE_MODEL = 'dbmdz/bert-base-italian-xxl-cased'
tokenizer = AutoTokenizer.from_pretrained(BPE_MODEL)
MAX_ID_TOKEN = len(tokenizer)
