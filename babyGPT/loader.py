
import os.path
import pickle
import spacy
from torch.utils.data import IterableDataset, DataLoader
import torch
import numpy as np
from typing import Sequence
from pathlib import Path
from config import selected_config as conf


# ╦ ╦╦ ╦╔═╗╔═╗╦═╗╔═╗╔═╗╦═╗╔═╗╔╦╗╔═╗╔╦╗╔═╗╦═╗╔═╗
# ╠═╣╚╦╝╠═╝║╣ ╠╦╝╠═╝╠═╣╠╦╝╠═╣║║║║╣  ║ ║╣ ╠╦╝╚═╗
# ╩ ╩ ╩ ╩  ╚═╝╩╚═╩  ╩ ╩╩╚═╩ ╩╩ ╩╚═╝ ╩ ╚═╝╩╚═╚═╝

MAX_DIST = 5 # Max distance between word and context
NEGATIVE_SAMPLE_PROB = 0.87   # Probability of a negative sample
MAX_LEN = 15_000 # Number of words in vocabulary
BATCH_SIZE = 200
# Skipping words  double the execution time
THRESHOLD = 3e-5 # Never skip below this threshold
BASE_SKIP = 1.9 # Increase for skipping more words (default is 1.2, always greater than 1)
OFFSET_SKIP = 0.025 # Decrease for skipping more words  (default is 0.00358)
ONLY_PREVIOUS = True # Context need to precede embedding

#======================================================

root = Path(__file__).resolve().parents[1]

file = str(root / 'dataset/wiki_it_ruby_processed.txt')
device = torch.device("cpu")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#======================================================

class TextChunkIterator:
    """
    Class for iterating small chunks of text of bigtext files

    It takes a (big) text file, break it in chunks of size `chunk_size` and index all chunks.
    It makes also a dictionary of occurrence for most frequent words

    This class is an iterator, so you can tou use it for iterating among the chunks
    """
    def __init__(self, filepath: str, chunks_positions:Sequence, **kwargs):
        """
        :param filepath: path of text file containing sentences
        :param chunks_positions: list containing all positions of each chunk in the form
                        [[start_chunk_1, end_chunk_2], [start_chunk_2, end_chunk_2], ...]
        """
        #self.nlp = spacy.load("it_core_news_sm")
        self.filepath = filepath
        self.chunks_positions = chunks_positions
        self._word_list = None
        self._vocab = None
        self._word_frequency = None


    @staticmethod
    def calculate_chunks_index(text_file, chunk_size=10000)-> Sequence[int]:
        """This is executed only one time. It iterates in the text and build a list of all positions
        where a new chunk start.
        Each chunk has AT LEAST chunk_size characters"""
        chunks_index= [0]
        cursor_position = 0
        for line in text_file:
            cursor_position += len(line)
            if cursor_position - chunks_index[-1] > chunk_size:
                chunks_index.append(cursor_position)
        # Add the last position of the text file, if it has not been already added
        if chunks_index[-1] != cursor_position:
            chunks_index.append(cursor_position)
        return chunks_index

    @staticmethod
    def build_chunks_index(filepath: str) -> None:
        """Generate an index file containing chunks positions with pickle"""
        with open(filepath, 'rb') as f:
            chunks_indexes = TextChunkIterator.calculate_chunks_index(f)
        with open(filepath + ".index", 'wb') as f:
            pickle.dump(chunks_indexes, f)

    @staticmethod
    def get_chunks_index(filepath)->Sequence[int]:
        """It returns the list of all chunks positions, using the index file, or generating it if
        it does not exist"""
        import os
        index_path = filepath + ".index"
        if not os.path.exists(index_path):
            print(
                "The index does not exists, creating one (the operation could take some time ...)")
            TextChunkIterator.build_chunks_index(filepath)
        with open(index_path, 'rb') as f:
            return pickle.load(f)

    def process_chunk(self, chunk: bytes) -> str:
        """Some minor modification to text, removing spaces and special characters"""
        chunk = chunk.decode(encoding='utf-8')
        chunk = chunk.replace('\r', '')
        chunk = chunk.replace('\n', ' ')
        chunk = chunk.replace('  ', ' ')
        chunk = chunk.strip()
        return chunk

    def __getitem__(self, item: int) -> str:
        """Return a chunk of text"""
        # retrieve chunk position
        start, end = self.chunks_positions[item]
        offset = end - start
        # read the chunk, do some processing and return it
        with open(self.filepath, 'rb') as f:
            f.seek(start)
            chunk = f.read(offset)
            chunk = self.process_chunk(chunk)
        return chunk


    def __len__(self):
        return len(self.chunks_positions)

    @staticmethod
    def split(filepath, seed=3, ratios=[0.75, 0.25], **kwargs) -> Sequence["TextChunkIterator"]:
        """Split into train and test (or more possibilities)"""
        import numpy as np

        np.random.seed(seed)
        #Start position of each chunk
        chunks_index = TextChunkIterator.get_chunks_index(filepath)
        # Start and end of each chunk
        chunks_positions = list(zip(chunks_index, chunks_index[1:]))
        np.random.shuffle(chunks_positions)
        # Get positions in the list of chunks corresponding to the ration given as parameters
        # Ex: ratio=[0.75, 0.25], chunks_positions=[[1,10], [10, 23], [23,38], [38,49],... ] of length 100
        #     --> ratio_index = [0, 75, 100]
        ratio_index = np.floor(np.cumsum(([0] + ratios)/np.sum(ratios)) * len(chunks_positions)).astype(int)
        # Split (shuffled) batch position list using ratio_index
        pos_list_splitted = [chunks_positions[i:j] for i,j in zip(ratio_index, ratio_index[1:])]
        # Return a list of TextChunkIterator instance corresponding to each value of ratios

        return [TextChunkIterator(filepath, pos_list_i, **kwargs) for pos_list_i in pos_list_splitted]

    def make_vocab(self, vocab_path, max_len=MAX_LEN):
        """Create (and save) a vocabulary containing max_len most common words"""
        from collections import Counter
        from tqdm import tqdm
        # count words in each chunk
        word_count = Counter()
        for chunk in tqdm(self):
            word_list = [str(token) for token in self.nlp.tokenizer(chunk) if (not token.is_punct and not token.is_digit)]
            word_count.update(word_list)
        # keep only more common words
        vocab = word_count.most_common(max_len)
        # save vocabulary
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)


    @property
    def vocab(self):
        """Return the vocabulary of most common words"""
        if self._vocab is None:
            self.load_vocab()
            self._vocab = {t[0]: i for i, t in enumerate(self._word_list)}
        return self._vocab

    @property
    def word_frequency(self):
        # Return frequency for each word in vocabulary
        if self._word_frequency is None:
            self.load_vocab()
            counts = [c for _, c in self._word_list]
            tot = sum(counts)
            freqs = [c/tot for c in counts]
            self._word_frequency = { t[0]: freq for t, freq in zip(self._word_list, freqs) }

        return self._word_frequency


    def load_vocab(self, force=False):
        """Load stored vocabulary"""
        vocab_path = self.filepath + '.vocab'
        if not os.path.exists(vocab_path) or force:
            print("Building a vocabulary. It could take some time ...")
            self._word_list = self.make_vocab(vocab_path)
        with open(vocab_path, 'rb') as f:
            self._word_list = pickle.load(f)

    def tensorize(self, word):
        """Convert a word to corresponding dummy tensor"""
        # TODO: Look for a built-in method on torch to apply on chunk for increasing performance
        #  of ~20%
        dummy = torch.zeros(len(self.vocab))
        # dummy = torch.zeros(1, device=self.device)
        word_index = self.vocab[word]
        dummy[word_index] = 1

        return dummy

class SentenceChunkIterator:
    """
    Class for iterating among sentences in a short text
    """

    def __init__(self, text, text_chunk_iterator, device=device):
        self.text_chunk_iterator = text_chunk_iterator
        #self.nlp = text_chunk_iterator.nlp
        self.text = text
        # TODO: add below to preprocessing
        # Below can be done in preprocessing maybe, it takes 8 ms for batch
        # Text is splitted in sentences

        # self.sentences = [[str(token) for token in sentence] for sentence in self.nlp(text).sents]
        self.sentences = text.split('.')
        self.device = device
        self.tokenizer = conf.TOKENIZER




    def sentence_generator(self):
        """A generator iterating a couple (word, context) label in the batch text"""
        # index of the currently processed sentence

        def tokenize(sentence: str) -> Sequence[str]:
            tokenized = self.tokenizer.encode(sentence)
            n_token = len(tokenized)
            # Mark sentence as too long, so it will be skipped
            if n_token > conf.MAX_N_TOKEN:
                return None, 0
            pad_tail = torch.ones(conf.MAX_N_TOKEN - n_token) * self.tokenizer.pad_token_id
            padded_tokenized = torch.cat([torch.tensor(tokenized), pad_tail], axis=0)
            return padded_tokenized.int(), n_token

        self.curr_sentence = 0

        # Iterate sentence among sentences
        while self.curr_sentence < len(self.sentences):
            sentence = self.sentences[self.curr_sentence]
            self.curr_sentence += 1
            tokenized, n_token = tokenize(sentence)
            # Skip sentences too long
            if n_token == 0:
                continue
            print(sentence)
            yield tokenized, n_token

class SentencesTextIterator:
    """Iterate sentences among the batches of the big text
    This class substantially merge TextBatchIterator and SentenceChunkIterator
    """

    def __init__(self, text_chunk_iterator, device=device):
        self.text_chunk_iterator = text_chunk_iterator
        self.device = device

    def __iter__(self):
        self.chunk_idx = 0
        next_chunk = SentenceChunkIterator(self.text_chunk_iterator[self.chunk_idx],
                                        self.text_chunk_iterator, device=self.device)
        self.sentence_iterator = next_chunk.sentence_generator()
        return self

    def __next__(self):

        try:
            next_sentence = next(self.sentence_iterator)
        except StopIteration:
            self.chunk_idx += 1
            if self.chunk_idx >= len(self.text_chunk_iterator):
                raise StopIteration
            else:
                next_batch = SentenceChunkIterator(self.text_chunk_iterator[self.chunk_idx],
                                                self.text_chunk_iterator)
                self.sentence_iterator = next_batch.sentence_generator()
                return self.__next__()

        return next_sentence



class LongTextDataset(IterableDataset):
    """Just a wrapper of SentencesTextIterator as Torch dataset"""
    def __init__(self, text_chunk_iterator, **kwargs):
        self.iterator = SentencesTextIterator(text_chunk_iterator, **kwargs)

    def __iter__(self):
        return self.iterator.__iter__()


class LongTextDataloader():
    """Remove excessive padding to batched samples """
    def __init__(self, *args, **kwargs):
        self.dataloader = DataLoader(*args, **kwargs)


    def __iter__(self):
        self.iter_dataloader = iter(self.dataloader)
        return self

    def __next__(self):
        batch, n_token = next(self.iter_dataloader)
        max_length = n_token.max()
        cropped_batch = batch[:, 0:max_length]
        return cropped_batch

train, test = TextChunkIterator.split(file, seed=4, ratios=[8, 2])
train_dataset = LongTextDataset(train, device=conf.DEVICE)
test_dataset = LongTextDataset(test, device=conf.DEVICE)
train_dataloader = LongTextDataloader(train_dataset, batch_size=conf.BATCH_SIZE)
test_dataloader = LongTextDataloader(test_dataset, batch_size=conf.BATCH_SIZE)

if __name__ == "__main__":

    # train_iterator = LongTextDataset(train)
    # for x in train_iterator:
    #     print(x)


    import time

    def batch_time():
        n = 1_000
        for i, (sentence) in zip(range(n + 101), train_dataloader):
            if i == 100:
                tic = time.time()
            if i == n + 100:
                tac = time.time()
            print(sentence)
            pass  # print(word, label)

        print(f"{1000 * (tac - tic) / n:.0f} ms")


    batch_time()