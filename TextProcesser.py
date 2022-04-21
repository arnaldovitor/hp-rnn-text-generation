import re
import string

import numpy as np


class TextProcesser:
    def __init__(self):
        self.text = None
        self.train = None
        self.val = None
        self.test = None


    def __clean(self, books_list):
        text = []
        for i in books_list:
            with open('harry_potter_books_corpora/Book{}.txt'.format(i)) as f:
                book = f.read()
                book = book.lower()
                book = re.sub('\n', '', book)

                for p in string.punctuation:
                    book = book.replace(p, '')
                for p in ['”', '—', '“', 'jk', 'rowling']:
                    book = book.replace(p, '')

                text += book.split(' ')

        self.text = text


    def __split_three(self, ratio):
        train_r, val_r, test_r = ratio
        assert (np.sum(ratio) == 1.0)
        indicies_for_splitting = [int(len(self.text) * train_r), int(len(self.text) * (train_r + val_r))]
        self.train, self.val, self.test = np.split(self.text, indicies_for_splitting)


    def run(self, books_list, ratio=[0.8, 0.1, 0.1]):
        self.__clean(books_list)
        self.__split_three(ratio)
        return self.train, self.val, self.test

