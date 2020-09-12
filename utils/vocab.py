import pickle
from collections import Counter

from transformers import BertTokenizer, RobertaTokenizer

from utils import SimpleTokenizer


class Vocabulary(object):
    """
    Wrapper class for vocabulary
    """

    def __init__(self, vocab_from_pretrained=None, do_lower_case=True):
        self.vocab_from_pretrained = vocab_from_pretrained
        self.do_lower_case = do_lower_case
        if vocab_from_pretrained:
            if "bert-" in vocab_from_pretrained:
                self.tokenizer = BertTokenizer.from_pretrained(
                    vocab_from_pretrained, do_lower_case=do_lower_case
                )
                self.word2idx = dict(self.tokenizer.vocab)
            elif "roberta-" in vocab_from_pretrained:
                self.tokenizer = RobertaTokenizer.from_pretrained(
                    vocab_from_pretrained, do_lower_case=do_lower_case
                )
                self.word2idx = {}
                self.idx2word = {}
                self._counter = Counter()
                self._size = 0

            self.idx2word = {v: k for k, v in self.word2idx.items()}
            self._counter = Counter()
            self._size = len(self.word2idx)
        else:
            self.tokenizer = SimpleTokenizer()
            self.word2idx = {}
            self.idx2word = {}
            self._counter = Counter()
            self._size = 0
        # Order is too important!!! Please, look at SimpleTokenizer if you want to change!
        self.add_word(self.tokenizer.pad_token)
        self.add_word(self.tokenizer.unk_token)
        self.add_word(self.tokenizer.cls_token)
        self.add_word(self.tokenizer.sep_token)

    def add_word(self, word):
        """
        Adds a token to the vocabulary
        :param word: (str) word to add to vocabulary
        :return: None
        """
        if word not in self.word2idx:
            if self.vocab_from_pretrained:
                if "roberta-" in self.vocab_from_pretrained:
                    self.idx2word[self.tokenizer.convert_tokens_to_ids(word)] = word
                    self.word2idx[word] = self.tokenizer.convert_tokens_to_ids(word)
            else:
                self.idx2word[self._size] = word
                self.word2idx[word] = self._size
            self._size += 1
        self._counter[word] += 1

    def add_text(self, text):
        """
        Splits text into tokens and adds to the vocabulary
        :param text: (str) text to add to vocabulary
        :return: None
        """
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        for token in tokens:
            self.add_word(token)

    def clean_text(self, text):
        """
        Cleans text for processing
        :param text: (str) text to be cleaned
        :return: (str) cleaned text
        """
        return text

    def tokenize(self, text):
        """
        Splits text into individual tokens
        :param text: (str) text to be tokenized
        :return: (list) list of tokens in text
        """
        return self.tokenizer.tokenize(text)

    def most_common(self, n):
        """
        Creates a new vocabulary object containing the n most frequent tokens from current vocabulary
        :param n: (int) number of most frequent tokens to keep
        :return: (Vocabulary) vocabulary containing n most frequent tokens
        """
        tmp = Vocabulary()

        for w in self._counter.most_common(n):
            tmp.add_word(w[0])
            tmp._counter[w[0]] = w[1]
        return tmp

    def load(self, path="vocab.pkl"):
        """
        Loads vocabulary from given path
        :param path: (str) path to pkl object
        :return: None
        """
        with open(path, "rb") as f:
            self.__dict__.clear()
            self.__dict__.update(pickle.load(f))
        print("\nVocabulary successfully loaded from [{}]\n".format(path))

    def save(self, path="vocab.pkl"):
        """
        Saves vocabulary to given path
        :param path: (str) path where vocabulary should be stored
        :return: None
        """
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print("\nVocabulary successfully stored as [{}]\n".format(path))

    def __len__(self):
        """
        Number of unique words in vocabulary
        """
        return self._size

    def __str__(self):
        s = "Vocabulary contains {} tokens\nMost frequent tokens:\n".format(self._size)
        for w in self._counter.most_common(10):
            s += "{} : {}\n".format(w[0], w[1])
        return s

    def __getitem__(self, item):
        """
        Returns the word corresponding to an id or and id corresponding to a word in the vocabulary.
        Return <unknown> if id/word is not present in the vocabulary
        """
        if isinstance(item, int):
            return self.idx2word[item]
        elif isinstance(item, str):
            if item in self.word2idx:
                return self.word2idx[item]
            else:
                return self.word2idx[self.tokenizer.unk_token]
        return None
