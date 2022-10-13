'''
Author:     Yunfan Long
Project:    DL4Jazz
Purpose:    LSTM Models for Music Generation
'''

from __future__ import print_function

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Dense, Activation, Dropout, Flatten, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
import numpy as np

''' Build a 2-layer LSTM from a training corpus '''
def build_model(corpus, val_indices, max_len, N_epochs=128, model=None):
    # number of different values or words in corpus
    N_values = len(set(corpus))

    # cut the corpus into semi-redundant sequences of max_len values
    step = 3
    sentences = []
    next_values = []
    for i in range(0, len(corpus) - max_len, step):
        sentences.append(corpus[i: i + max_len])
        next_values.append(corpus[i + max_len])
    print('nb sequences:', len(sentences))

    # transform data into binary matrices
    X = np.zeros((len(sentences), max_len, N_values), dtype=np.bool)
    y = np.zeros((len(sentences), N_values), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, val in enumerate(sentence):
            X[i, t, val_indices[val]] = 1
        y[i, val_indices[next_values[i]]] = 1
    
    if model=='lstm-vae':
        model, _, _ = lstm_vae(input_dim=N_values, 
                               timesteps=max_len, 
                               batch_size=N_epochs, 
                               intermediate_dim=128, 
                               latent_dim=64, 
                               epsilon_std=1.) 
    
    elif model=='lstm':
        # build a 2 stacked LSTM
        # default
        model = lstm(input_dim=N_values, timesteps=max_len)

    model.fit(X, y, batch_size=128, epochs=N_epochs)

    return model


def lstm(input_dim, timesteps):
    # build a 2 stacked LSTM
    # default
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(timesteps, input_dim)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model