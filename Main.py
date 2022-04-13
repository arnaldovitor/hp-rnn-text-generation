from TextProcesser import TextProcesser
from DataGen import DataGen
from TrainingSection import TrainingSection
from GRU import GRUModel

import tensorflow as tf

num_words = 2048
batch_size = 16
num_steps = 16
num_hiddens = 1024
vocab_size = 2048
num_epochs = 1000
lr = 1e-2

# Process text object.
tp = TextProcesser()
train, val, test = tp.run(books_list=[1, 2, 3, 4, 5, 6, 7],
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

GRUModel = GRUModel(vocab_size=vocab_size,
                    embedding_dim=vocab_size,
                    rnn_units=num_hiddens)

ts = TrainingSection(net=GRUModel,
                     train_data=train_data,
                     val_data=val_data,
                     vocab=vocab)

ts.run(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
       updater=tf.keras.optimizers.SGD(lr),
       epochs=num_epochs,
       prefix='wizard is',
       num_preds=8,
       token_param=tokenizer)
