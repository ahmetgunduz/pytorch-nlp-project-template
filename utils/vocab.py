import pickle
from collections import Counter


class Vocabulary(object):
    """
    Wrapper class for vocabulary
    """

    def __init__(self):
        self._word2idx = {}
        self._idx2word = {}
        self._counter = Counter()
        self._size = 0
        self._punctuation2token = {';': "<semicolon>",
                                   ':': "<colon>",
                                   "'": "<inverted_comma>",
                                   '"': "<quotation_mark>",
                                   ',': "<comma>",
                                   '\n': "<new_line>",
                                   '!': "<exclamation_mark>",
                                   '-': "<hyphen>",
                                   '--': "<hyphens>",
                                   '.': "<period>",
                                   '?': "<question_mark>",
                                   '(': "<left_paren>",
                                   ')': "<right_paren>",
                                   '♪': "<music_note>",
                                   '[': "<left_square>",
                                   ']': "<right_square>",
                                   "’": "<inverted_comma>",
                                   }
        self.add_text('<pad>')
        self.add_text('<unknown>')

    def add_word(self, word):
        """
        Adds a token to the vocabulary
        :param word: (str) word to add to vocabulary
        :return: None
        """
        word = word.lower()
        if word not in self._word2idx:
            self._idx2word[self._size] = word
            self._word2idx[word] = self._size
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
        text = text.lower().strip()
        for key, token in self._punctuation2token.items():
            text = text.replace(key, ' {} '.format(token))
        text = text.strip()
        while '  ' in text:
            text = text.replace('  ', ' ')
        return text

    def tokenize(self, text):
        """
        Splits text into individual tokens
        :param text: (str) text to be tokenized
        :return: (list) list of tokens in text
        """
        return text.split(' ')

    def set_vocab(self, vocab):
        self._word2idx = {}
        self._idx2word = {}
        self._counter = Counter()
        self._size = 0
        self.add_text('<pad>')
        self.add_text('<unknown>')
        for word in vocab:
            self.add_word(word)

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

    def load(self, path='vocab.pkl'):
        """
        Loads vocabulary from given path
        :param path: (str) path to pkl object
        :return: None
        """
        with open(path, 'rb') as f:
            self.__dict__.clear()
            self.__dict__.update(pickle.load(f))
        print("\nVocabulary successfully loaded from [{}]\n".format(path))

    def save(self, path='vocab.pkl'):
        """
        Saves vocabulary to given path
        :param path: (str) path where vocabulary should be stored
        :return: None
        """
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print("\nVocabulary successfully stored as [{}]\n".format(path))

    def add_punctuation(self, text):
        """
        Replces punctuation tokens with corresponding characters
        :param text: (str) text to process
        :return: text with punctuation tokens replaced with characters
        """
        for key, token in self._punctuation2token.items():
            text = text.replace(token, ' {} '.format(key))
        text = text.strip()
        while '  ' in text:
            text = text.replace('  ', ' ')
        text = text.replace(' :', ':')
        text = text.replace(" ' ", "'")
        text = text.replace("[ ", "[")
        text = text.replace(" ]", "]")
        text = text.replace(" .", ".")
        text = text.replace(" ,", ",")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" ’ ", "’")
        return text

    def __len__(self):
        """
        Number of unique words in vocabulary
        """
        return self._size

    def __str__(self):
        s = "Vocabulary contains {} tokens\nMost frequent tokens:\n".format(
            self._size)
        for w in self._counter.most_common(10):
            s += "{} : {}\n".format(w[0], w[1])
        return s

    def __getitem__(self, item):
        """
        Returns the word corresponding to an id or and id corresponding to a word in the vocabulary.
        Return <unknown> if id/word is not present in the vocabulary
        """
        if isinstance(item, int):
            return self._idx2word[item]
        elif isinstance(item, str):
            if item in self._word2idx:
                return self._word2idx[item]
            else:
                return self._word2idx['<unknown>']
        return None
