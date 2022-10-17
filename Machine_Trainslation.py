#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 09:08:59 2021

@author: kali
"""

import collections
import os
import numpy as np
import keras
from tensorflow.keras import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
import pandas as pd


df =pd.read_csv('/home/kali/Documents/Wrk/zindi/Machine_Translation/Train.csv')
FR_FON = df.loc[df['Target_Language'] == 'Fon']
FR_EWE = df.loc[df['Target_Language'] == 'Ewe']

fr_fon=FR_FON.drop(['Target_Language','Target'],axis=1)
fon=FR_FON.drop(['French','Target_Language'],axis=1)
fr_ewe = FR_EWE.drop(['Target_Language','Target'],axis=1)
ewe=FR_EWE.drop(['French','Target_Language'],axis=1)

french_ewe_sentences = fr_ewe.iloc[:,1]
ewe_sentences=ewe.iloc[:,1]

#Uses Tokenizer class to return token objetcs
def tokenize(x):
    x_tk = Tokenizer(char_level = False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen = length, padding = 'post')

def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)# Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_x = preprocess_x.reshape(*preprocess_x.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk 

preproc_ewe, preproc_french_ewe, ewe_tokenizer, french_ewe_tokenizer =\
    preprocess(ewe_sentences, french_ewe_sentences)
    
max_french_ewe_length = preproc_french_ewe.shape[1]
max_ewe_length = preproc_ewe.shape[1]
french_ewe_vocab_size = len(french_ewe_tokenizer.word_index)
ewe_vocab_size = len(ewe_tokenizer.word_index)
    
def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])    

tmp_x = pad(preproc_french_ewe, max_french_ewe_length)
temp_y=pad(preproc_ewe, max_french_ewe_length)
tmp_x = tmp_x.reshape((-1, 107, 1))# Train the neural network

def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 1e-3
    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences = True)(input_seq)
    logits = TimeDistributed(Dense(french_vocab_size))(rnn)
    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
    return model

simple_rnn_model = simple_model(
    tmp_x.shape,
    max_french_ewe_length,
    french_ewe_vocab_size,
    ewe_vocab_size+1)

simple_rnn_model.fit(tmp_x, temp_y, batch_size=1024, epochs=10, validation_split=0.2)# Print prediction(s)


def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
  
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size,output_dim=128,input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256,return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256,return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))
    learning_rate = 0.005
    
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
    
    return model




model = model_final(tmp_x.shape,
                        preproc_french_sentences.shape[1],
                        len(english_tokenizer.word_index)+1,
                        len(french_tokenizer.word_index)+1)
    
model.fit(tmp_x, preproc_french_sentences, batch_size = 1024, epochs = 17, validation_split = 0.2)
    
