"""
This module implements some utility for iterating phrases among a long text.
It supports cached indexation of the text, cached word occurrence vocabulary and
train / text split.

There are three principal classes:

- TextChunkIterator:   For iterating small chunks of text in a huge text file.
                       Text file is split in small chunks, each chunk position is cached in a pickle
                       file, word occurrency is calculated and cache too. Also, chunks are
                       shuffled and splitted in train/test.

- NGramChunkIterator:  For iterating couples of near words in a chunk of text. Words are converted to
                       dummy tensor using vocabulary built during text indexation, and more frequent
                       words are skipped. Also, with a ratio given by NEGATIVE_SAMPLE_PROB ,
                       random couple of words are returned as negative samples

- NGramTextIterator:   It is just a wrapper of TextChunkIterator and NGramChunkIterator, for iterating
                       couples of near words in the full huge text

There is also the class LongTextDataset, which is just NGramTextIterator as pytorch IterableDataset


"""

import os.path
import pickle
import spacy
from torch.utils.data import IterableDataset, DataLoader
import torch
import numpy as np
from typing import Sequence
from pathlib import Path

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
        self.nlp = spacy.load("it_core_news_sm")
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


class NGramChunkIterator:
    """
    Class for iterating among Ngram in a short text
    a Ngram is a cuple word-context at a short distance in the same sentence

    Given a chunk of text, it iterate couples word, context (embedded as dummy tensor) in the
    sentences (detected using spacy). Negative samples are added.
    """

    def __init__(self, text, text_chunk_iterator, max_dist=MAX_DIST, only_previous=ONLY_PREVIOUS, device=device):
        self.text_chunk_iterator = text_chunk_iterator
        self.nlp = text_chunk_iterator.nlp
        self.text = text
        # TODO: add below to preprocessing
        # Below can be done in preprocessing maybe, it takes 8 ms for batch
        # Text is splitted in sentences
        self.sentences = [[str(token) for token in sentence] for sentence in self.nlp(text).sents]
        # self.sentences = [text.split()]

        self.vocab = text_chunk_iterator.vocab
        self.word_frequency = text_chunk_iterator.word_frequency
        self.len_vocab = len(self.vocab)
        # self.window is a list of integers defining admitted distances between word and context
        if only_previous:
            self.window = [x for x in range(-max_dist, 0)]
        else:
            self.window = [x for x in range(-max_dist, max_dist + 1) if x != 0]
        self.device = device


    def skip_too_frequent(self, word):
        """Skip words with too many occurrence"""
        freq = self.word_frequency[word]
        if freq > THRESHOLD:
            ratio = np.clip(OFFSET_SKIP / freq * BASE_SKIP ** (np.log(freq)), 0, 1)
            r = np.random.random()
            if r > ratio:
                return True
        return False


    def tensorize(self, *args, **kwargs):
        """Convert a word to a tensor"""
        return self.text_chunk_iterator.tensorize(*args, **kwargs)


    def word_context_generator(self):
        """A generator iterating a couple (word, context) label in the batch text"""
        # index of the currently processed sentence
        self.curr_sentence = 0
        # index of the currently processed word
        self.curr_word = 0
        # index of the currently processed context
        self.curr_context = 0
        previous_word = ""
        # Iterate sentence among sentences
        while self.curr_sentence < len(self.sentences):
            sentence = self.sentences[self.curr_sentence]
            # iterate among words in the sentence
            if self.curr_word < len(sentence):
                word = str(sentence[self.curr_word])
            else:
                self.curr_word = 0
                self.curr_sentence += 1
                self.curr_context = 0
                continue
            # if the word is not in the vocabulary, skip
            if word not in self.vocab:
                self.curr_word += 1
                continue
            # if the word is too frequent, maybe skip
            if self.skip_too_frequent(word):
                # Skip a word with increasing probability if very frequent
                self.curr_word += 1
                continue
            # Iterate context using window variable
            if self.curr_context < len(self.window):
                context_pos = self.window[self.curr_context] + self.curr_word
            else:
                self.curr_word += 1
                self.curr_context = 0
                continue
            # deal the case where context calculated position is before the beginning of the sentence
            if context_pos < 0:
                self.curr_context += 1
                continue
            # deal the case where context calculated position is after the end of the sentence
            if context_pos >= len(sentence):
                self.curr_word += 1
                self.curr_context = 0
                continue
            # if context is not in the vocabulary thn skip
            context = str(sentence[context_pos])
            if context not in self.vocab:
                self.curr_context += 1
                continue
            # if the context is too frequent, maybe skip
            if self.skip_too_frequent(context):
                self.curr_context += 1
                continue

            if previous_word != word:
                # caching tensor obtained from embedded word for not calculating it twice
                embedded_word = self.tensorize(word)
                previous_word = word

            # Give some negative samples before the positive one

            while np.random.random() < NEGATIVE_SAMPLE_PROB:
                # negative sample
                # keep the word and choose a random context
                # TODO: optimize random choice for reducing time of 1-2 ms par batch
                # TODO: chose world with probability depending on frequency. Maybe using old
                #  context word from a rolling batch ?
                # Choose a random word
                word_index = np.random.randint(0, self.len_vocab)
                # embedded_context = torch.zeros(1, device=self.device)
                embedded_context = torch.zeros(self.len_vocab)
                embedded_context[word_index] = 1
                label = torch.tensor([0])
                features = torch.stack([embedded_word, embedded_context])
                yield features, label

            #positive sample
            label = torch.tensor([1])
            self.curr_context += 1
            # embedded_context = torch.zeros(1)
            embedded_context = self.tensorize(context)
            # Performing so many stack decrease performance
            features = torch.stack([embedded_word, embedded_context])
            yield features, label


class NGramTextIterator:
    """Iterate NGrams among the batches of the big text

    This class sostantially merge TextBatchIterator and NGramBatchIterator

    So you declare it using a TextBatchIterator instance, and you get an iterator for word,
    context among the big text file
    """

    def __init__(self, text_chunk_iterator, device=device):
        self.text_chunk_iterator = text_chunk_iterator
        self.device = device

    def __iter__(self):
        self.chunk_idx = 0
        next_chunk = NGramChunkIterator(self.text_chunk_iterator[self.chunk_idx],
                                        self.text_chunk_iterator, max_dist=MAX_DIST, device=self.device)
        self.ngram_iterator = next_chunk.word_context_generator()
        return self

    def __next__(self):

        try:
            next_ngram = next(self.ngram_iterator)
        except StopIteration:
            self.chunk_idx += 1
            if self.chunk_idx >= len(self.text_chunk_iterator):
                raise StopIteration
            else:
                next_batch = NGramChunkIterator(self.text_chunk_iterator[self.chunk_idx],
                                                self.text_chunk_iterator, max_dist=MAX_DIST)
                self.ngram_iterator = next_batch.word_context_generator()
                return self.__next__()

        return next_ngram


class LongTextDataset(IterableDataset):
    """Just a wrapper of NGramTextIterator as Torch dataset"""
    def __init__(self, text_chunk_iterator, **kwargs):
        self.iterator = NGramTextIterator(text_chunk_iterator, **kwargs)

    def __iter__(self):
        return self.iterator.__iter__()


train, test = TextChunkIterator.split(file, seed=4, ratios=[8, 2])
vocab = train.vocab
train_dataset = LongTextDataset(train, device=device)
test_dataset = LongTextDataset(test, device=device)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

if __name__ == "__main__":



    import matplotlib.pyplot as plt
    import seaborn as sns

    # fig, ax = plt.subplots(1, 1)
    #
    # sns.ecdfplot(wc.values(), ax=ax)
    # ax.set_xscale('log')
    # fig.show()

    # 25 ms for batch with 200 of batch size (without skip)
    # 80 ms for batch with 200 of batch size (with skip)
    # 34 ms for batch with 200 of batch size (with skip) and negative samples (0.8)
    # 50 ms for the old model on cpu
    # AFTER OPTIMIZATION/DEBUGGING: 16 ms (1500 words)
    # 28 ms (15 000 words)
    import time

    def batch_time():
        n = 1_000
        for i, (features, label) in zip(range(n+101), train_dataloader):
            if i==100:
                tic = time.time()
            if i == n+100:
                tac = time.time()
            features = features.to(device)
            label = label.to(device)
            pass
            # print(word, label)

        print(f"{1000*(tac - tic) / n:.0f} ms")
    batch_time()





