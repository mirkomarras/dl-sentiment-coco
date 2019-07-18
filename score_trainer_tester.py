from keras.layers import RepeatVector, dot, LSTM, Permute, merge, TimeDistributed, Bidirectional, Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, GaussianNoise, Activation, MaxoutDense, Embedding, SimpleRNN, MaxoutDense
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.engine.topology import Layer
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn import preprocessing
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.callbacks import *
from sklearn.metrics import *
import pandas as pd
import numpy as np
import argparse
import time
import sys
import csv
import os

def getClass2(value):
    split = 2.5
    values = [0, 1]
    if value <= split:
        return values[0]
    return values[1]

def getClass5(value):
    splits = [1.0, 2.0, 3.0, 4.0]
    values = [0, 1, 2, 3, 4]
    if value <= splits[0]:
        return values[0]
    elif value <= splits[1]:
        return values[1]
    elif value <= splits[2]:
        return values[2]
    elif value <= splits[3]:
        return values[3]
    return values[4]

def getClass10(value):
    splits = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if value <= splits[0]:
        return values[0]
    elif value <= splits[1]:
        return values[1]
    elif value <= splits[2]:
        return values[2]
    elif value <= splits[3]:
        return values[3]
    elif value <= splits[4]:
        return values[4]
    if value <= splits[5]:
        return values[5]
    elif value <= splits[6]:
        return values[6]
    elif value <= splits[7]:
        return values[7]
    elif value <= splits[8]:
        return values[8]
    return values[9]

def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class Attention(Layer):

    def __init__(self, W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],), initializer=self.init, name='{}_W'.format(self.name), regularizer=self.W_regularizer, constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],), initializer='zero', name='{}_b'.format(self.name), regularizer=self.b_regularizer, constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def getModel(embedding_matrix, max_words, max_len, embedding_dim):
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_len))
    model.add(GaussianNoise(0.2))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2, implementation=1)))
    model.add(Attention())
    model.add(Dropout(0.3))
    model.add(MaxoutDense(100, W_constraint=maxnorm(2)))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    return model

def main():
    parser = argparse.ArgumentParser(description='DL Sentiment Training')
    parser.add_argument('--traintest_file', dest='traintest_file', default='data/traintest_course_comments.csv', type=str, action='store', help='Train and test course comments CSV file')
    parser.add_argument('--comment_field', dest='comment_field', default='learner_comment', type=str, action='store', help='Field title for comments in CSV file')
    parser.add_argument('--score_field', dest='score_field', default='learner_rating', type=str, action='store', help='Field title for scores in CSV file')
    parser.add_argument('--max_len', dest='max_len', default=500, type=int, action='store', help='Max number of words in a comment')
    parser.add_argument('--n_classes', dest='n_classes', default=2, type=int, action='store', help='Number of prediction classes')
    parser.add_argument('--embs_dir', dest='embs_dir', default='embeddings/generic', type=str, action='store', help='Directory containing the embeddings files')
    parser.add_argument('--n_epochs', dest='n_epochs', default=20, type=int, action='store', help='Number of training epochs')
    parser.add_argument('--batch_size', dest='batch_size', default=256, type=int, action='store', help='Training batch size')
    parser.add_argument('--n_fold', dest='n_fold', default=10, type=int, action='store', help='Number of validation folds')

    args = parser.parse_args()
    reviews = pd.read_csv(args.traintest_file)
    texts = reviews[args.comment_field].tolist()
    labels = reviews[args.score_field].tolist()

    print('* LOADING DATA')
    print('Sample comment and score', labels[0], texts[0])

    max_len = args.max_len

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    all_words = [word for word, i in word_index.items()]
    max_words = len(word_index) + 1
    print('Found %s unique words.' % len(word_index))

    data = pad_sequences(sequences, maxlen=max_len)
    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    print('* PRE-PROCESSING DATA')
    skf = StratifiedKFold(n_splits=args.n_fold)

    print('Scaling score values')
    if args.n_classes == 10:
        discrete_labels = np.array([getClass10(x) for x in labels])
    elif args.n_classes == 5:
        discrete_labels = np.array([getClass5(x) for x in labels])
    else:
        discrete_labels = np.array([getClass2(x) for x in labels])

    print('Shuffling data')
    s = np.arange(data.shape[0])
    np.random.shuffle(s)
    data = data[s]
    discrete_labels = discrete_labels[s]

    print('* LOADING EMBEDDINGS AND MODELS')
    print('Found', len(os.listdir(args.embs_dir)), 'embeddings files')

    for embs_file in os.listdir(args.embs_dir):
        print('Loading embeddings from', embs_file)
        embedding_dim = int(embs_file.split('_')[1])
        embedding_path = os.path.join(args.embs_dir,embs_file)

        embeddings_index = {}
        f = open(embedding_path)
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:-1], dtype='float32')
            if word in all_words:
                embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

        embedding_matrix = np.zeros((max_words, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if i < max_words:
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

        for id_fold, (train_index, test_index) in enumerate(skf.split(data, discrete_labels)):
            print('Training model for fold', id_fold)
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = discrete_labels[train_index], discrete_labels[test_index]

            mcheck = ModelCheckpoint(os.path.join('models', 'class' + str(args.n_classes), embs_file.replace('.txt', '_fold' + str(id_fold) + '_model.h5')), monitor='val_loss', save_best_only=True)
            model = getModel(embedding_matrix, max_words, max_len, embedding_dim)
            model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy', 'mse'])

            for epoch in range(args.n_epochs):
                model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch+1, initial_epoch=epoch, batch_size=args.batch_size, shuffle=True, callbacks=[mcheck], verbose=1)
                y_pred = model.predict(X_test)
                y_pred = [item for sublist in y_pred.tolist() for item in sublist]
                y_pred_rounded = [int(round(x)) for x in y_pred]

                with open(os.path.join('results', 'class' + str(args.n_classes), embs_file.replace('.txt', '_fold' + str(id_fold) + '_results.txt')), mode='a') as result_file:
                    result_writer = csv.writer(result_file, delimiter=',')
                    result_writer.writerow([epoch, mean_squared_error(y_test.tolist(), y_pred), mean_absolute_error(y_test.tolist(), y_pred), accuracy_score(y_test.tolist(), y_pred_rounded), precision_score(y_test.tolist(), y_pred_rounded, average='weighted'), recall_score(y_test.tolist(), y_pred_rounded, average='weighted'), f1_score(y_test.tolist(), y_pred_rounded, average='weighted')])

if __name__ == "__main__":
    main()
