import os.path
import pickle




def index_text_file(file, batch_size=1000):
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


def generate_file_index(filepath):
    """Generate a index file for line numbers"""

    with open(filepath, 'rb') as f:
        file_index = index_text_file(f)
    with open(filepath + ".index", 'wb') as f:
            pickle.dump(file_index, f)


def open_index(filepath):
    """It return the list of new lines associated to filepath. Open the index file, and if it does
    not exists, it creates one"""
    import os
    index_path = filepath + ".index"
    if not os.path.exists(index_path):
        print("The index does not exists, creating one (the operation could take some time ...)")
        generate_file_index(filepath)
    with open(index_path, 'rb') as f:
        return pickle.load(f)

class TextBatchIterator:
    """Class for iterating in small chunks of bigtext files"""
    def __init__(self, filepath, pos_list):
        import spacy

        self.nlp = spacy.load("it_core_news_sm")
        self.filepath = filepath
        self.pos_list = pos_list
        self._vocab = None

    def process_batch(self, batch):
        batch = batch.decode(encoding='utf-8')
        batch = batch.replace('\r', '')
        batch = batch.replace('\n', ' ')
        batch = batch.replace('  ', ' ')
        batch = batch.strip()
        return batch

    def __getitem__(self, item):
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
    def split(filepath, seed=3, ratios=[0.75, 0.25]):
        """Split into train and test (or more possibilities)"""
        import numpy as np

        np.random.seed(seed)
        index = open_index(filepath)
        pos_list = list(zip(index, index[1:]))
        np.random.shuffle(pos_list)
        ratio_index = np.floor(np.cumsum(([0] + ratios)/np.sum(ratios)) * len(pos_list)).astype(int)
        pos_list_splitted = [pos_list[i:j] for i,j in zip(ratio_index, ratio_index[1:])]

        return [TextBatchIterator(filepath, pos_list_i) for pos_list_i in pos_list_splitted]

    def make_vocab(self, vocab_path, max_len=1500):
        """Make a vacabulary"""
        from collections import Counter
        from tqdm import tqdm

        word_count = Counter()
        for b in tqdm(self):
            l = [str(token) for token in self.nlp.tokenizer(b)  if (not token.is_punct and not token.is_digit)]
            word_count.update(l)
        vocab = word_count.most_common(max_len)
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)


    @property
    def vocab(self):
        if self._vocab is None:
            self.load_vocab()
        return {t[0]: i for i, t in  enumerate(self._vocab)}

    @property
    def word_frequency(self):
        if self._vocab is None:
            self.load_vocab()
        counts = [c for _, c in self._vocab]
        tot = sum(counts)
        freqs = [c/tot for c in counts]

        return {t[0]: freq for t, freq in  zip(self._vocab, freqs)}


    def load_vocab(self, force=False):
        vocab_path = self.filepath + '.vocab'
        if not os.path.exists(vocab_path) or force:
            print("Building a vocabulary. It could take some time ...")
            self._vocab = self.make_vocab(vocab_path)
        with open(vocab_path, 'rb') as f:
            self._vocab = pickle.load(f)

class NGramTextIterator:
    """Iterate NGrams among the batches of the big text"""
    def __init__(self, text_batch_iterator):
        self.text_batch_iterator = text_batch_iterator


    def __iter__(self):
        self.nbatch = 0
        next_batch = self.text_batch_iterator[self.nbatch]
        self.ngram_iterator = NGramBatchIterator(next_batch, self.text_batch_iterator).__iter__()
        return self

    def __next__(self):
        try:
            next_ngram = self.ngram_iterator.__next__()
        except StopIteration:
            self.nbatch += 1
            if self.nbatch >= len(self.text_batch_iterator):
                raise StopIteration
            else:
                next_batch = self.text_batch_iterator[self.nbatch]
                self.ngram_iterator = NGramBatchIterator(next_batch,
                                                               self.text_batch_iterator).__iter__()
                return self.__next__()

        return next_ngram


class NGramBatchIterator:
    """Class for iterating among Ngram in a short text"""
    def __init__(self, text, text_batch_iterator, max_dist=2):
        self.text_batch_iterator = text_batch_iterator
        self.nlp = text_batch_iterator.nlp
        self.sentences = [i for i in self.nlp(text).sents]

        self.vocab = text_batch_iterator.vocab
        self.window = [x for x in range(-max_dist, max_dist + 1) if x != 0]


    def __iter__(self):
        self.curr_sentence = 0
        self.curr_word = 0
        self.curr_context = 0
        return self


    def __next__(self):
        if self.curr_sentence < len(self.sentences):
            sentence = self.sentences[self.curr_sentence]
        else:
            raise StopIteration


        if self.curr_word < len(sentence):
            word = str(sentence[self.curr_word])
        else:
            self.curr_word = 0
            self.curr_sentence += 1
            self.curr_context = 0
            return self.__next__()

        if word not in self.vocab:
            #TODO: optimize with a sorted vocabulary
            self.curr_word += 1
            return self.__next__()

        if self.curr_context < len(self.window):
            context_pos = self.window[self.curr_context] + self.curr_word
        else:
            self.curr_word += 1
            self.curr_context = 0
            return self.__next__()

        if context_pos < 0:
            self.curr_context += 1
            return self.__next__()

        if context_pos >= len(sentence):
            self.curr_word += 1
            self.curr_context = 0
            return self.__next__()

        context = str(sentence[context_pos])
        if context not in self.vocab:
            #TODO: optimize with a sorted vocabulary
            self.curr_context += 1
            return self.__next__()

        self.curr_context += 1

        return word, context


if __name__ == "__main__":

    file = 'dataset/wiki_it_processed.txt'
    train, test = TextBatchIterator.split(file, seed=4, ratios=[8, 2])
    vocab = train.vocab
    print(vocab)
    print(train.word_frequency)

    batch = train[0]
    it = NGramTextIterator(train)




