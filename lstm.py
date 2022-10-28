'''
Author:     Yunfan Long
Project:    DL4Jazz
Purpose:    LSTM Models for Music Generation
'''

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector, Bidirectional
from keras.layers import Dense, Activation, Dropout, Flatten, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras.regularizers import l1, l2
from keras import objectives
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

''' Build a 2-layer LSTM from a training corpus '''
def build_model(corpus, val_indices, max_len, N_epochs=128, model_choice=None):
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
    
    
    if model_choice=='lstm':
        # default
        model = lstm(input_dim=N_values, timesteps=max_len)
        plot_model(model, to_file='lstm.png')
        history = model.fit(X, y, batch_size=128, epochs=N_epochs)
    
    elif model_choice=='vae-lstm':
        model, _, _ = vae_lstm(input_dim=N_values, 
                               timesteps=max_len, 
                               batch_size=N_epochs,
                               intermediate_dim=128, 
                               latent_dim=64, 
                               epsilon_std=0.5) 
        plot_model(model, to_file='vae-lstm.png')
        history = model.fit(X, X, batch_size=128, epochs=N_epochs)

    elif model_choice=='bi-lstm':
        model = bi_lstm(input_dim=N_values, timesteps=max_len)
        plot_model(model, to_file='bi-lstm.png')
        history = model.fit(X, y, batch_size=128, epochs=N_epochs)

    else:
        raise Exception

    return model


def lstm(input_dim, timesteps):
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

def bi_lstm(input_dim, timesteps):
    model = Sequential() #1
    model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(timesteps, input_dim))))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(610, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(610, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(610, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(610, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(610, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(610, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.2))

    model.add(Dense(input_dim, kernel_regularizer=l1(1e-3)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', )
    # model.fit(X, y, batch_size=128, nb_epoch=N_epochs)
    return model
    

def vae_lstm(input_dim, timesteps, batch_size, intermediate_dim, latent_dim, epsilon_std=1.):
    x = Input(shape=(timesteps, input_dim, ))

    # LSTM encoding
    h = LSTM(intermediate_dim, return_sequences=True)(x)

    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)
    print('This is mu:', K.int_shape(z_mean), 'latent_dim', latent_dim)
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(timesteps, latent_dim, ),
                                  mean=0., stddev=epsilon_std)
        z = K.mean(z_mean + z_log_sigma * epsilon, axis=1)
        return z

    z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean, z_log_sigma])
    
    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)
    
    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)
    encoder.summary()

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    generator.summary()
    
    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        loss = xent_loss + kl_loss
        return loss

    vae.compile(optimizer='adam', loss=vae_loss)
    
    return vae, encoder, generator