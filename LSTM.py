import tensorflow as tf


class LSTMModel(tf.keras.Model):
    def __init__(self, vocab_size, rnn_units):
        super().__init__(self)
        self.vocab_size = vocab_size
        self.lstm1 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, dropout=0.2)
        self.lstm2 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, dropout=0.2)
        self.lstm3 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, dropout=0.2)
        # self.lstm4 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, dropout=0.2)
        # self.lstm5 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, dropout=0.2)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = tf.one_hot(x, self.vocab_size)

        if states is None:
            states = self.lstm1.get_initial_state(x)

        x, *states = self.lstm1(x, initial_state=states, training=training)
        x, *states = self.lstm2(x, initial_state=states, training=training)
        x, *states = self.lstm3(x, initial_state=states, training=training)
        # x, states = self.lstm4(x, initial_state=states, training=training)
        # x, states = self.lstm5(x, initial_state=states, training=training)

        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


