import math

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.utils.generic_utils import Progbar
from matplotlib import pyplot as plt


class TrainingSection:
    def __init__(self, net, train_data, val_data, vocab):
        self.net = net
        self.train_data = train_data
        self.val_data = val_data
        self.vocab = vocab
        self.history = {'train-loss': [], 'train-ppl': [],  'val-loss': [], 'val-ppl': []}
        self.loss = None
        self.updater = None

    def __grad_clipping(self, grads, theta):
        theta = tf.constant(theta, dtype=tf.float32)
        new_grad = []
        for grad in grads:
            if isinstance(grad, tf.IndexedSlices):
                new_grad.append(tf.convert_to_tensor(grad))
            else:
                new_grad.append(grad)
        norm = tf.math.sqrt(
            sum((tf.reduce_sum(grad ** 2)).numpy() for grad in new_grad))
        norm = tf.cast(norm, tf.float32)
        if tf.greater(norm, theta):
            for i, grad in enumerate(new_grad):
                new_grad[i] = grad * theta / norm
        else:
            new_grad = new_grad
        return new_grad

    def __train_step(self, x, y):
        with tf.GradientTape(persistent=True) as g:
            y_hat = self.net(x)
            y = tf.one_hot(y, self.net.vocab_size)
            y = tf.cast(y, tf.float32)
            l = self.loss(y, y_hat)

        grads = g.gradient(l, self.net.trainable_variables)
        grads = self.__grad_clipping(grads, 1)
        self.updater.apply_gradients(zip(grads, self.net.trainable_variables))
        return l

    def __val_step(self, x, y):
        y_hat = self.net(x)
        y = tf.one_hot(y, self.net.vocab_size)
        y = tf.cast(y, tf.float32)
        l = self.loss(y, y_hat)
        return l

    def predict(self, prefix, num_preds, token_param):
        outputs = [self.vocab[prefix.split()[0]]]
        get_input = lambda: tf.reshape(tf.constant([outputs[-1]]), (1, 1)).numpy()

        for y in prefix.split()[1:]:
            _ = self.net(get_input())
            outputs.append(self.vocab[y])

        for _ in range(num_preds):
            y = self.net(get_input())
            y = tf.squeeze(y)
            y = int(tf.argmax(y))
            outputs.append(y)

        return token_param.sequences_to_texts([outputs])

    def run(self, loss, updater, epochs, prefix, num_preds, token_param):
        self.loss = loss
        self.updater = updater

        for epoch in range(epochs):
            print("\nTRAINING: epoch {}/{}".format(epoch + 1, epochs))
            pb_train = Progbar(len(self.train_data) * self.train_data.batch_size, stateful_metrics=['train-loss', 'train-ppl'])

            total_train_loss = 0.0
            for i, (x, y) in enumerate(self.train_data):
                total_train_loss += self.__train_step(x, y)
                total_train_ppl = math.exp(total_train_loss.numpy() / len(self.train_data))

                if i % 2 == 0:
                    pb_train.update(i * self.train_data.batch_size,
                                    values=[('train-loss', total_train_loss / len(self.train_data)),  ('train-ppl', total_train_ppl / len(self.train_data))])

            self.history['train-loss'].append(total_train_loss / len(self.train_data))
            self.history['train-ppl'].append(total_train_ppl / len(self.train_data))

            print("\nVALIDATING: epoch {}/{}".format(epoch + 1, epochs))
            pb_val = Progbar(len(self.val_data) * self.val_data.batch_size, stateful_metrics=['val-loss', 'val-ppl'])

            total_val_loss = 0.0
            for i, (x, y) in enumerate(self.val_data):
                total_val_loss += self.__val_step(x, y)
                total_val_ppl = math.exp(total_val_loss.numpy() / len(self.val_data))

                if i % 2 == 0:
                    pb_val.update(i * self.val_data.batch_size,
                                  values=[('val-loss', total_val_loss / len(self.val_data)), ('val-ppl', total_val_ppl / len(self.val_data))])

            self.history['val-loss'].append(total_val_loss / len(self.val_data))
            self.history['val-ppl'].append(total_val_ppl / len(self.val_data))

            print('\nPREDICT: {}'.format(self.predict(prefix, num_preds, token_param)))

            plt.plot(self.history['train-loss'], label='train-loss')
            plt.plot(self.history['val-loss'], label='val-loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('loss-decay.png')
            plt.clf()

            self.net.save_weights('weights.h5')

        return self.history

