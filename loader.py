import os.path
import pickle
import spacy
from torch.utils.data import IterableDataset, DataLoader
import torch
import numpy as np


# ╦ ╦╦ ╦╔═╗╔═╗╦═╗╔═╗╔═╗╦═╗╔═╗╔╦╗╔═╗╔╦╗╔═╗╦═╗╔═╗
# ╠═╣╚╦╝╠═╝║╣ ╠╦╝╠═╝╠═╣╠╦╝╠═╣║║║║╣  ║ ║╣ ╠╦╝╚═╗
# ╩ ╩ ╩ ╩  ╚═╝╩╚═╩  ╩ ╩╩╚═╩ ╩╩ ╩╚═╝ ╩ ╚═╝╩╚═╚═╝

MAX_DIST = 2 # Max distance between word and context
NEGATIVE_SAMPLE_PROB = 0.87   # Probability of a negative sample
MAX_LEN = 15_000 # Number of words in vocabulary
BATCH_SIZE = 200

# Skipping words  double the execution time
BASE_SKIP = 1.2 # Increase for skipping fewer words (default is 1.2)
OFFSET_SKIP = 8 # Increase for skipping fewer words  (default is 8)

#======================================================

file = 'dataset/wiki_it_processed.txt'
device = torch.device("cpu")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TextBatchIterator:
    """
    Class for iterating small chunks of (iterabl) text of bigtext files

    It takes a (big) text file, break it in chunks of size `batch_size` and index all chunks.
    It makes also a dictionary of occurrence for most frequent words

    This class is an iterator, so you can tou use it for iterate among the chunks
    """
    def __init__(self, filepath, pos_list, device=device):


        self.nlp = spacy.load("it_core_news_sm")
        self.filepath = filepath
        self.pos_list = pos_list
        self._word_list = None
        self._vocab = None
        self._word_frequency = None
        self.device = device

    @staticmethod
    def index_text_file(file, batch_size=10000):
        """Read in the file once and build a list of batches"""
        batch_indexes = [0]
        offset = 0
        for line in file:
            offset += len(line)
            if offset - batch_indexes[-1] > batch_size:
                batch_indexes.append(offset)
        if batch_indexes[-1] != offset:
            batch_indexes.append(offset)
        return batch_indexes

    def generate_file_index(self):
        """Generate a index file for line numbers"""

        with open(self.filepath, 'rb') as f:
            file_index = self.index_text_file(f)
        with open(self.filepath + ".index", 'wb') as f:
            pickle.dump(file_index, f)

    @staticmethod
    def open_index(filepath):
        """It return the list of new lines associated to filepath. Open the index file, and if it does
        not exists, it creates one"""
        import os
        index_path = filepath + ".index"
        if not os.path.exists(index_path):
            print(
                "The index does not exists, creating one (the operation could take some time ...)")
            TextBatchIterator.generate_file_index()
        with open(index_path, 'rb') as f:
            return pickle.load(f)

    def process_batch(self, batch: bytes) -> "NGramBatchIterator":
        """Some minor modification to text, and create a NGramBatchIterator object"""
        batch = batch.decode(encoding='utf-8')
        batch = batch.replace('\r', '')
        batch = batch.replace('\n', ' ')
        batch = batch.replace('  ', ' ')
        batch = batch.strip()

        return batch

    def __getitem__(self, item: int) -> "NGramBatchIterator":
        """Return a NGramBatchIterator object"""
        start, end = self.pos_list[item]
        offset = end - start
        with open(self.filepath, 'rb') as f:
            f.seek(start)
            batch = f.read(offset)
            batch = self.process_batch(batch)
        return batch


    def __len__(self):
        return len(self.pos_list)

    @staticmethod
    def split(filepath, seed=3, ratios=[0.75, 0.25], **kwargs):
        """Split into train and test (or more possibilities)"""
        import numpy as np

        np.random.seed(seed)
        index = TextBatchIterator.open_index(filepath)
        pos_list = list(zip(index, index[1:]))
        np.random.shuffle(pos_list)
        ratio_index = np.floor(np.cumsum(([0] + ratios)/np.sum(ratios)) * len(pos_list)).astype(int)
        pos_list_splitted = [pos_list[i:j] for i,j in zip(ratio_index, ratio_index[1:])]

        return [TextBatchIterator(filepath, pos_list_i, **kwargs) for pos_list_i in pos_list_splitted]

    def make_vocab(self, vocab_path, max_len=MAX_LEN):
        """Make a vocabulary"""
        from collections import Counter
        from tqdm import tqdm

        word_count = Counter()
        for b in tqdm(self):
            l = [str(token) for token in self.nlp.tokenizer(b) if (not token.is_punct and not token.is_digit)]
            word_count.update(l)
        vocab = word_count.most_common(max_len)
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)


    @property
    def vocab(self):
        if self._vocab is None:
            self.load_vocab()
            self._vocab = {t[0]: i for i, t in enumerate(self._word_list)}
        return self._vocab

    @property
    def word_frequency(self):
        if self._word_frequency is None:
            self.load_vocab()
            counts = [c for _, c in self._word_list]
            tot = sum(counts)
            freqs = [c/tot for c in counts]
            self._word_frequency = { t[0]: freq for t, freq in zip(self._word_list, freqs) }
        return self._word_frequency


    def load_vocab(self, force=False):
        vocab_path = self.filepath + '.vocab'
        if not os.path.exists(vocab_path) or force:
            print("Building a vocabulary. It could take some time ...")
            self._word_list = self.make_vocab(vocab_path)
        with open(vocab_path, 'rb') as f:
            self._word_list = pickle.load(f)

    def tensorize(self, word):
        """Convert a word to a tensor"""
        # TODO: Look for a built-in method on torch to apply on batch for increasing performance
        #  of ~20%
        dummy = torch.zeros(len(self.vocab))
        # dummy = torch.zeros(1, device=self.device)
        word_index = self.vocab[word]
        dummy[word_index] = 1
        return dummy



class NGramBatchIterator:
    """
    Class for iterating among Ngram in a short text

    Given a chunk of text, it iterate couples word, context (embedded as dummy tensor) in the
    sentences (detected using spacy). Negative samples are added.
    """

    def __init__(self, text, text_batch_iterator, max_dist=MAX_DIST):
        self.text_batch_iterator = text_batch_iterator
        self.nlp = text_batch_iterator.nlp
        self.text = text
        # TODO: add below to preprocessing
        # Below can be done in preprocessing maybe, it takes 8 ms for batch
        self.sentences = [[str(token) for token in sentence] for sentence in self.nlp(text).sents]
        # self.sentences = [text.split()]

        self.vocab = text_batch_iterator.vocab
        self.word_frequency = text_batch_iterator.word_frequency
        self.len_vocab = len(self.vocab)
        self.window = [x for x in range(-max_dist, max_dist + 1) if x != 0]
        self.device = text_batch_iterator.device


    def skip_too_frequent(self, word):
        """Skip words with too many occurrency"""
        freq = self.word_frequency[word]
        if freq > 1e-3:
            ratio = 1e-3 / freq * BASE_SKIP**(np.log(freq)+OFFSET_SKIP)
            r = np.random.random()
            if r > ratio:
                return True
        return False




    def tensorize(self, *args, **kwargs):
        """Convert a word to a tensor"""
        return self.text_batch_iterator.tensorize(*args, **kwargs)


    def word_context_generator(self):
        """A generator iterating a couple (word, context) label in the batch text"""
        self.curr_sentence = 0
        self.curr_word = 0
        self.curr_context = 0
        old_word = ""
        while self.curr_sentence < len(self.sentences):
            sentence = self.sentences[self.curr_sentence]

            if self.curr_word < len(sentence):
                word = str(sentence[self.curr_word])
            else:
                self.curr_word = 0
                self.curr_sentence += 1
                self.curr_context = 0
                continue

            if word not in self.vocab:
                self.curr_word += 1
                continue

            if self.skip_too_frequent(word):
                # Skip a word with increasing probability if very frequent
                self.curr_word += 1
                continue

            if self.curr_context < len(self.window):
                context_pos = self.window[self.curr_context] + self.curr_word
            else:
                self.curr_word += 1
                self.curr_context = 0
                continue

            if context_pos < 0:
                self.curr_context += 1
                continue

            if context_pos >= len(sentence):
                self.curr_word += 1
                self.curr_context = 0
                continue

            context = str(sentence[context_pos])
            if context not in self.vocab:
                self.curr_context += 1
                continue

            if self.skip_too_frequent(context):
                self.curr_context += 1
                continue

            if old_word != word:
                # caching embedded word
                embedded_word = self.tensorize(word)
                old_word = word

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

    def __init__(self, text_batch_iterator):
        self.text_batch_iterator = text_batch_iterator
        self.device = text_batch_iterator.device

    def __iter__(self):
        self.nbatch = 0
        next_batch = NGramBatchIterator(self.text_batch_iterator[self.nbatch],
                                                        self.text_batch_iterator, max_dist=MAX_DIST)
        self.ngram_iterator = next_batch.word_context_generator()
        return self

    def __next__(self):

        try:
            next_ngram = next(self.ngram_iterator)
        except StopIteration:
            self.nbatch += 1
            if self.nbatch >= len(self.text_batch_iterator):
                raise StopIteration
            else:
                next_batch = NGramBatchIterator(self.text_batch_iterator[self.nbatch],
                                                        self.text_batch_iterator, max_dist=MAX_DIST)
                self.ngram_iterator = next_batch.word_context_generator()
                return self.__next__()

        return next_ngram


class LongTextDataset(IterableDataset):
    """Just a wrapper of NGramTextIterator as Torch dataset"""
    def __init__(self, text_batch_iterator, **kwargs):
        self.iterator = NGramTextIterator(text_batch_iterator, **kwargs)

    def __iter__(self):
        return self.iterator.__iter__()


train, test = TextBatchIterator.split(file, seed=4, ratios=[8, 2], device=device)
vocab = train.vocab
train_dataset = LongTextDataset(train)
test_dataset = LongTextDataset(test)
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





