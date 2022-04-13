import random

import tensorflow as tf


class DataGen:
    def __init__(self, batch_size, text, num_steps, num_words):
        self.batch_size = batch_size
        self.text = text
        self.num_steps = num_steps
        self.num_words = num_words
        self.data_iter_fn = self.__seq_data_iter_random
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words)
        self.tokenizer.fit_on_texts(text)
        self.vocab = self.tokenizer.word_index
        self.corpus = [x for y in self.tokenizer.texts_to_sequences(text) for x in y]
        self.n = (len(self.corpus) - 1) // num_steps

    def __seq_data_iter_random(self, corpus, batch_size, num_steps):
        corpus = corpus[random.randint(0, num_steps - 1):]
        num_subseqs = (len(corpus) - 1) // num_steps
        initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
        random.shuffle(initial_indices)

        def data(pos):
            return corpus[pos:pos + num_steps]

        num_batches = num_subseqs // batch_size

        for i in range(0, batch_size * num_batches, batch_size):
            initial_indices_per_batch = initial_indices[i:i + batch_size]
            X = [data(j) for j in initial_indices_per_batch]
            Y = [data(j + 1) for j in initial_indices_per_batch]

            yield tf.constant(X), tf.constant(Y)

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

    def __len__(self):
        return self.n // self.batch_size

    def get_vocab(self):
        return self.vocab

    def get_tokenizer(self):
        return self.tokenizer
