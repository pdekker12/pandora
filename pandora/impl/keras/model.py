#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np

from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Input, Bidirectional, Conv1D, Flatten, Dropout
from keras.layers import Dense, Activation, concatenate, RepeatVector
from keras.layers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.models import model_from_json
import keras.backend as K
from keras.models import load_model

from pandora.impl.base_model import BaseModel


class KerasModel(BaseModel):
    """
    Keras Implementation of the Base Model
    """
    CONFIG_KEY = "Keras"

    def __init__(self, settings, build=True):
        self.settings = settings
        self.settings.nb_context_tokens = settings.nb_left_tokens + settings.nb_right_tokens

        if build:
            self._build()

    def print_summary(self):
        print(self.model.summary())

    def _build(self):
        """
        Function that builds/compiles the actual model based
        on the specified inputs and outputs.
        """

        inputs, outputs, subnets = [], [], []

        if self.settings.include_token:
            # add input layer:
            token_input, token_subnet = self._build_token_subnet()
            inputs.append(token_input), subnets.append(token_subnet)

        if self.settings.include_context:
            context_input, context_subnet = self._build_context_subnet()
            inputs.append(context_input), subnets.append(context_subnet)

        # combine subnets:
        if len(subnets) > 1:
            joined = concatenate(subnets, name='joined')
        else:
            joined = Activation('linear', name='joined')(subnets[0])

        if self.settings.include_lemma:
            lemma_label = self._build_lemma_decoder(joined)
            outputs.append(lemma_label)

        if self.settings.include_pos:
            pos_label = self._build_pos_decoder(joined)
            outputs.append(pos_label)

        if self.settings.include_morph:
            morph_label = self._build_morph_decoder(joined)
            outputs.append(morph_label)

        loss_dict = {}
        if self.settings.include_lemma:
            loss_dict['lemma_out'] = 'categorical_crossentropy'
        if self.settings.include_pos:
            loss_dict['pos_out'] = 'categorical_crossentropy'
        if self.settings.include_morph:
            if self.settings.include_morph == 'label':
                loss_dict['morph_out'] = 'categorical_crossentropy'
            elif self.settings.include_morph == 'multilabel':
                loss_dict['morph_out'] = 'binary_crossentropy'

        self.model = Model(inputs=inputs, outputs=outputs)
        if self.settings.focus_repr == 'convolutions':  # TODO: fix this?
            self.model.compile(optimizer='Adam', loss=loss_dict)
        else:
            self.model.compile(optimizer='Adam', loss=loss_dict)

    def _build_token_subnet(self):
        token_input = Input(shape=(self.settings.max_token_len,),
                            dtype='int32', name='focus_in')

        token_embed = Embedding(input_dim=self.settings.nb_token_chars,
                                output_dim=self.settings.char_embed_dim,
                                input_length=self.settings.max_token_len,
                                name='char_embedding')(token_input)

        if self.settings.focus_repr == 'recurrent':
            curr_enc_out = None
            for i in range(self.settings.nb_encoding_layers):
                if i == 0:      # first layer
                    curr_input = token_embed
                else:
                    curr_input = curr_enc_out
                if i == (self.settings.nb_encoding_layers - 1):  # last layer
                    token_subnet = Bidirectional(
                        LSTM(units=self.settings.nb_dense_dims,
                             return_sequences=False,
                             activation='tanh',
                             name='final_focus_encoder'),
                        merge_mode='sum')(curr_input)
                else:           # hidden layer
                    curr_enc_out = Bidirectional(
                        LSTM(units=self.settings.nb_dense_dims,
                             return_sequences=True,
                             activation='tanh',
                             name='encoder_'+str(i+1)),
                        merge_mode='sum')(curr_input)

        elif self.settings.focus_repr == 'convolutions':
            token_subnet = Conv1D(
                input_shape=(self.settings.max_token_len, self.settings.nb_token_chars),
                activation='relu',
                name='focus_conv',
                filters=self.settings.nb_filters,
                kernel_size=self.settings.filter_length,
                padding='valid',
                strides=1,
                kernel_initializer='glorot_uniform')(token_embed)
            token_subnet = Flatten(name='focus_flat')(token_subnet)
            token_subnet = Dropout(
                self.settings.dropout_level, name='focus_dropout1')(token_subnet)
            token_subnet = Dense(
                self.settings.nb_dense_dims, name='focus_dense')(token_subnet)
            token_subnet = Dropout(
                self.settings.dropout_level, name='focus_dropout2')(token_subnet)
            token_subnet = Activation(
                'relu', name='final_focus_encoder')(token_subnet)

        else:
            raise ValueError('Parameter `focus_repr` not understood: ' +
                             'use "recurrent" or "convolutions".')

        return token_input, token_subnet

    def _build_context_subnet(self):
        context_input = Input(shape=(self.settings.nb_context_tokens,),
                              dtype='int32', name='context_in')
        context_subnet = Embedding(input_dim=self.settings.nb_train_tokens,
                                   output_dim=self.settings.nb_embedding_dims,
                                   weights=self.settings.pretrained_embeddings,
                                   input_length=self.settings.nb_context_tokens,
                                   name='context_embedding')(context_input)
        context_subnet = Flatten(name='context_flatten')(context_subnet)
        context_subnet = Dropout(
            self.settings.dropout_level, name='context_dropout')(context_subnet)
        context_subnet = Activation(
            'relu', name='context_relu')(context_subnet)
        context_subnet = Dense(
            self.settings.nb_dense_dims, name='context_dense1')(context_subnet)
        context_subnet = Dropout(
            self.settings.dropout_level, name='context_dropout2')(context_subnet)
        context_subnet = Activation('relu', name='context_out')(context_subnet)
        return context_input, context_subnet

    def _build_lemma_decoder(self, joined):
        if self.settings.include_lemma == 'generate':

            repeat = RepeatVector(
                self.settings.max_lemma_len, name='encoder_repeat')(joined)

            curr_out = None
            for i in range(self.settings.nb_encoding_layers):
                if i == 0:
                    curr_input = repeat
                else:
                    curr_input = curr_out

                if i == (self.settings.nb_encoding_layers - 1):
                    output_name = 'final_focus_decoder'
                else:
                    output_name = 'decoder_'+str(i + 1)

                curr_out = Bidirectional(
                    LSTM(units=self.settings.nb_dense_dims,
                         return_sequences=True,
                         activation='tanh',
                         name=output_name),
                    merge_mode='sum')(curr_input)
                # add lemma decoder
            lemma_label = TimeDistributed(
                Dense(self.settings.nb_lemma_chars),
                name='lemma_dense')(curr_out)
            lemma_label = Activation(
                'softmax', name='lemma_out')(lemma_label)

        elif self.settings.include_lemma == 'label':
            lemma_label = Dense(self.settings.nb_lemmas, name='lemma_dense1')(joined)
            lemma_label = Dropout(
                self.settings.dropout_level, name='lemma_dense_dropout1')(lemma_label)
            lemma_label = Activation('softmax', name='lemma_out')(lemma_label)

        return lemma_label

    def _build_pos_decoder(self, joined):
        pos_label = Dense(self.settings.nb_tags, name='pos_dense1')(joined)
        pos_label = Dropout(
            self.settings.dropout_level, name='pos_dense_dropout1')(pos_label)
        pos_label = Activation('softmax', name='pos_out')(pos_label)
        return pos_label

    def _build_morph_decoder(self, joined):
        if self.settings.include_morph == 'label':
            morph_label = Dense(self.settings.nb_dense_dims,
                                activation='relu',
                                name='morph_dense1')(joined)
            morph_label = Dropout(self.settings.dropout_level,
                                  name='morph_dense_dropout1')(morph_label)
            morph_label = Dense(self.settings.nb_dense_dims,
                                activation='relu',
                                name='morph_dense2')(morph_label)
            morph_label = Dropout(self.settings.dropout_level,
                                  name='morph_dense_dropout2')(morph_label)
            morph_label = Dense(self.settings.nb_morph_cats,
                                activation='relu',
                                name='morph_dense3')(morph_label)
            morph_label = Dropout(self.settings.dropout_level,
                                  name='morph_dense_dropout3')(morph_label)
            morph_label = Activation('softmax', name='morph_out')(morph_label)

        elif self.include_morph == 'multilabel':
            morph_label = Dense(self.settings.nb_dense_dims,
                                activation='relu',
                                name='morph_dense1')(joined)
            morph_label = Dropout(self.settings.dropout_level,
                                  name='morph_dense_dropout1')(morph_label)
            morph_label = Dense(self.settings.nb_dense_dims,
                                activation='relu',
                                name='morph_dense2')(morph_label)
            morph_label = Dropout(self.settings.dropout_level,
                                  name='morph_dense_dropout2')(morph_label)
            morph_label = Dense(self.settings.nb_morph_cats,
                                activation='relu',
                                name='morph_dense3')(morph_label)
            morph_label = Dropout(self.settings.dropout_level,
                                  name='morph_dense_dropout3')(morph_label)
            morph_label = Activation('tanh', name='morph_out')(morph_label)

        return morph_label

    def adjust_lr(self, adjust_rate=0.5):  # FIXME: does Adam need lr update?
        old_lr = K.get_value(self.model.optimizer.lr)
        new_lr = np.float32(old_lr * adjust_rate)
        K.set_value(self.model.optimizer.lr, new_lr)
        print('\t- Lowering learning rate > was:', old_lr, ', now:', new_lr)

    def epoch(self, train_in, train_out):
        self.model.fit(train_in, train_out,
                       epochs=1,
                       shuffle=True,
                       batch_size=self.settings.batch_size)

    def predict(self, input_data, batch_size=None):
        preds = self.model.predict(
            input_data, batch_size=settings.batch_size)
        if isinstance(preds, np.ndarray):
            out = [preds]
        else:
            out = preds
        
        labels = []
        if self.include_lemma:
            labels.append('lemma_out')
        if self.include_pos:
            labels.append('pos_out')
        if self.include_morph:
            labels.append('morph_out')
        
        assert len(out) == len(labels)
        return {k: v for (k, v) in zip(labels, out)}

    @staticmethod
    def load(model_dir,
             include_lemma=True, include_pos=True, include_morph=True,
             batch_size=50):
        # load model and weights:
        with open(os.path.join(model_dir, 'model_architecture.json'), 'r') as f:
            model = model_from_json(f.read())
        model.load_weights(os.sep.join((model_dir, 'model_weights.hdf5')))

        loss_dict = {}
        if include_lemma:
            model.include_lemma = include_lemma
            loss_dict['lemma_out'] = 'categorical_crossentropy'
        if include_pos:
            model.include_pos = include_pos
            loss_dict['pos_out'] = 'categorical_crossentropy'
        if include_morph:
            model.include_morph = include_morph
            if include_morph == 'label':
                loss_dict['morph_out'] = 'categorical_crossentropy'
            elif include_morph == 'multilabel':
                loss_dict['morph_out'] = 'binary_crossentropy'

        model.compile(optimizer='Adam', loss=loss_dict)
        keras_model = KerasModel(include_lemma=settings.include_lemma,
                                 include_pos=include_pos,
                                 include_morph=include_morph,
                                 batch_size=batch_size,
                                 build=False)
        keras_model.model = model

        return keras_model

    def save(self, model_dir):
        """
        Serializes the model.

        Parameters
        ===========
        model_dir : str
            Path to the model directory where the model should be saved.
        """
        # save architecture:
        json_string = self.model.to_json()
        with open(os.sep.join((model_dir, 'model_architecture.json')), 'wb') as f:
            f.write(json_string.encode())
        # save weights:
        self.model.save_weights(os.sep.join((model_dir, 'model_weights.hdf5')), overwrite=True)
