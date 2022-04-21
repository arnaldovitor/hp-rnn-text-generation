from TextProcesser import TextProcesser
from DataGen import DataGen
from TrainingSection import TrainingSection
from GRU import GRUModel
from ERNN import ERNNModel
from RNN import RNNModel
from DRNN import DRNNModel
from LSTM import LSTMModel

import tensorflow as tf

num_words = vocab_size = 512
batch_size = 32
num_steps = 16
num_hiddens = 1024
num_epochs = 100
lr = 1e-4

# Process text object.
tp = TextProcesser()
train, val, test = tp.run(books_list=[4],
                          ratio=[0.7, 0.1, 0.2])

# Train iterator.
train_data = DataGen(batch_size=batch_size,
                     text=train,
                     num_steps=num_steps,
                     num_words=num_words)

vocab = train_data.get_vocab()
tokenizer = train_data.get_tokenizer()

# Val iterator.
val_data = DataGen(batch_size=batch_size,
                   text=val,
                   num_steps=num_steps,
                   num_words=num_words)

ERNNModel = ERNNModel(vocab_size=vocab_size,
                    rnn_units=num_hiddens)

ts = TrainingSection(net=ERNNModel,
                     train_data=train_data,
                     val_data=val_data,
                     vocab=vocab)

ts.run(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
       updater=tf.keras.optimizers.Adam(lr),
       epochs=num_epochs,
       prefix='harry is',
       num_preds=8,
       token_param=tokenizer)
