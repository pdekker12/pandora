#!/usr/bin/env python
# -*- coding: utf-8

import os
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pandora.impl.base_model import BaseModel

from pandora.impl.pytorch.encoder import (
    RNNEncoder, ConvEncoder, BottleEncoder)
from pandora.impl.pytorch.decoder import AttentionalDecoder, LinearDecoder
from pandora.impl.pytorch.utils import Optimizer, BatchIterator, Progbar
from pandora.impl.pytorch import utils
from pandora.utils import PAD


class PyTorchModel(nn.Module, BaseModel):
    """
    Pytorch model.

    Parameters
    ===========
    method : str
        Optimizer method to use. One of Adam, Adagrad, Adadelta, SGD, RMSprop.
    gpu : bool
        Whether to run the model on the gpu or not.
    """
    CONFIG_KEY = "PyTorch"

    def __init__(self, token_len=None, token_char_vector_dict=None,
                 lemma_len=None, lemma_char_vector_dict=None,
                 nb_encoding_layers=None, nb_dense_dims=None, nb_tags=None,
                 nb_morph_cats=None, nb_lemmas=None, char_embed_dim=50,
                 nb_train_tokens=None, nb_context_tokens=None,
                 nb_embedding_dims=None, pretrained_embeddings=None,
                 include_token=True, include_context=True,
                 include_lemma=True, include_pos=True, include_morph=True,
                 nb_filters=100, filter_length=3, focus_repr='recurrent',
                 batch_size=None, dropout_level=0.15):
        self.token_len = token_len  # not used
        self.token_char_vector_dict = token_char_vector_dict
        self.nb_encoding_layers = nb_encoding_layers
        self.nb_dense_dims = nb_dense_dims
        self.lemma_len = lemma_len
        self.lemma_char_vector_dict = lemma_char_vector_dict
        self.nb_tags = nb_tags
        self.nb_morph_cats = nb_morph_cats
        self.nb_lemmas = nb_lemmas
        self.nb_train_tokens = nb_train_tokens
        self.nb_context_tokens = nb_context_tokens
        self.nb_embedding_dims = nb_embedding_dims
        self.pretrained_embeddings = pretrained_embeddings
        self.include_token = include_token
        self.include_context = include_context
        self.include_lemma = include_lemma
        self.include_pos = include_pos
        self.include_morph = include_morph
        self.nb_filters = nb_filters
        self.filter_length = filter_length
        self.focus_repr = focus_repr
        self.dropout_level = dropout_level
        self.char_embed_dim = char_embed_dim
        self.batch_size = batch_size
        self.joined_dim = 0
        if self.include_token:
            self.joined_dim = self.nb_dense_dims
        if self.include_context:
            self.joined_dim += self.nb_dense_dims
        super(PyTorchModel, self).__init__()

        # gpu
        self.gpu = False
        if torch.cuda.is_available():
            self.gpu = True

        # build subnets and losses
        self.token_embeddings, self.token_encoder = None, None
        self.context_embeddings, self.context_encoder = None, None
        self.lemma_decoder = None
        self.pos_decoder = None
        self.morph_decoder = None
        self.pos_loss, self.lemma_loss, self.morph_loss = None, None, None

        if self.include_token:
            self._build_token_subnet()

        if self.include_context:
            self._build_context_subnet()

        if self.include_lemma:
            self._build_lemma_decoder()
            self._build_lemma_loss()

        if self.include_pos:
            self._build_pos_decoder()
            self._build_pos_loss()

        if self.include_morph:
            self._build_morph_decoder()
            self._build_morph_loss()

        self.optimizer = Optimizer(self.parameters(), 'Adam', lr=0.001)

    def print_summary(self):
        # print model
        print(self)
        print()

        # print trainable parameters
        trainable, nontrainable = 0, 0
        for param in self.parameters():
            if param.requires_grad:
                trainable += param.nelement()
            else:
                nontrainable += param.nelement()

        print("* Number of trainable parameters: {}".format(trainable))
        print("* Number of non trainable parameters: {}".format(nontrainable))
        print("* Total parameters {}".format(trainable + nontrainable))

    def _build_lemma_loss(self):
        if self.include_lemma == 'generate':
            # weight down loss on padding
            lemma_weight = torch.ones(len(self.lemma_char_vector_dict))
            lemma_weight[self.lemma_char_vector_dict[PAD]] = 0
            self.lemma_loss = nn.NLLLoss(weight=lemma_weight)

        else:
            self.lemma_loss = nn.NLLLoss()

    def _build_pos_loss(self):
        self.pos_loss = nn.NLLLoss()

    def _build_morph_loss(self):
        if self.include_morph == 'label':
            self.morph_loss = nn.NLLLoss()
        else:
            self.morph_loss = nn.BCELoss()

    def _build_token_subnet(self):
        # embeddings
        self.token_embeddings = nn.Embedding(
            len(self.token_char_vector_dict),
            self.char_embed_dim)

        # init embeddings
        utils.init_embeddings(self.token_embeddings)

        # encoder
        if self.focus_repr == 'recurrent':
            self.token_encoder = RNNEncoder(
                num_layers=self.nb_encoding_layers,
                input_size=self.char_embed_dim,
                hidden_size=self.nb_dense_dims,
                dropout=self.dropout_level,
                merge_mode='concat')

        elif self.focus_repr == 'convolutions':

            if self.include_lemma and self.include_lemma == 'generate':
                if self.nb_filters != self.nb_dense_dims:
                    raise ValueError(
                        "Using convolutional encoding with generated lemmas "
                        "needs same number of filters and dense dimensions")

            self.token_encoder = ConvEncoder(
                in_channels=self.char_embed_dim,
                out_channels=self.nb_filters,
                kernel_size=self.filter_length,
                output_size=self.nb_dense_dims,
                token_len=self.token_len)

        else:
            raise ValueError('Parameter `focus_repr` not understood: ' +
                             'use "recurrent" or "convolutions".')

    def _build_context_subnet(self):
        self.context_embeddings = nn.Embedding(
            self.nb_train_tokens, self.nb_embedding_dims)

        # load pretrained embeddings
        if self.pretrained_embeddings is not None:
            weight = torch.from_numpy(np.array(self.pretrained_embeddings))
            self.context_embeddings.weight.data.copy_(weight)

        self.context_encoder = BottleEncoder(
            self.nb_embedding_dims, self.nb_dense_dims,
            seq_len=self.nb_context_tokens, dropout=self.dropout_level)

    def _build_lemma_decoder(self):
        if self.include_lemma == 'generate':
            self.lemma_decoder = AttentionalDecoder(
                char_dict=self.lemma_char_vector_dict,
                hidden_size=self.nb_dense_dims,
                char_embed_dim=self.char_embed_dim,
                include_context=self.include_context)

            # tie embeddings
            lemma_emb_size = self.lemma_decoder.embeddings.weight.size()
            token_emb_size = self.token_embeddings.weight.size()
            if lemma_emb_size == token_emb_size:
                self.lemma_decoder.embeddings.weight = \
                    self.token_embeddings.weight
            else:
                print("Lemma embedding size and token embedding size are not the same")  # Would it be an error ?

        elif self.include_lemma == 'label':
            if self.include_context:
                in_dim = self.joined_dim
            else:
                in_dim = self.nb_dense_dims
            self.lemma_decoder = LinearDecoder(
                in_dim, self.nb_lemmas, include_context=self.include_context)

        else:
            raise ValueError(
                "include_lemma must be either `generate` or `label`")

    def _build_pos_decoder(self):
        self.pos_decoder = nn.Sequential(
            nn.Linear(self.joined_dim, self.nb_tags),
            nn.Dropout(self.dropout_level),
            nn.LogSoftmax())

        utils.init_sequential_linear(self.pos_decoder)

    def _build_morph_decoder(self):
        if self.include_morph == 'label':
            self.morph_decoder = nn.Sequential(
                # morph_dense1
                nn.Linear(self.joined_dim, self.nb_dense_dims),
                nn.ReLU(),
                nn.Dropout(self.dropout_level),
                # morph_dense2
                nn.Linear(self.nb_dense_dims, self.nb_dense_dims),
                nn.ReLU(),
                nn.Dropout(self.dropout_level),
                nn.Linear(self.nb_dense_dims, self.nb_morph_cats),
                nn.ReLU(),  # TODO: perhaps remove ReLU before softmax
                nn.Dropout(self.dropout_level),
                nn.LogSoftmax())

        elif self.include_morph == 'multilabel':
            self.morph_decoder = nn.Sequential(
                # morph_dense1
                nn.Linear(self.joined_dim, self.nb_dense_dims),
                nn.ReLU(),
                nn.Dropout(self.dropout_level),
                # morph_dense2
                nn.Linear(self.nb_dense_dims, self.nb_dense_dims),
                nn.ReLU(),
                nn.Dropout(self.dropout_level),
                # morph_dense3
                nn.Linear(self.nb_dense_dims, self.nb_morph_cats),
                nn.Dropout(self.dropout_level),
                nn.Tanh())

        utils.init_sequential_linear(self.morph_decoder)

    def move_to_gpu(self, gpu=True):
        self.gpu = gpu
        if gpu:
            self.cuda()
        else:
            self.cpu()

    def adjust_lr(self, adjust_rate=0.5):
        if self.optimizer.method == 'SGD':
            for param_group in self.optimizer.optim.param_groups:
                param_group['lr'] *= adjust_rate

    def forward(self, train_in, train_out):
        """
        General function that computes both train and test model outputs.

        Parameters
        ===========
        train_in : dict with model inputs
        train_out : dict with model outputs (training) or None (testing)
        """
        token_out, context_out, token_context, joined = None, None, None, []
        if self.include_token:
            # (batch x token_len x emb_dim)
            token_out = self.token_embeddings(train_in['focus_in'])
            token_out = token_out.transpose(0, 1)
            # (batch x hidden), (seq_len x batch x hidden) where seq_len
            # is either the rnn seq_len, or the convolutional seq_len
            token_out, token_context = self.token_encoder(token_out)
            joined.append(token_out)

        if self.include_context:
            # (batch x seq_len x emb_dim)
            context_out = self.context_embeddings(train_in['context_in'])
            context_out = F.dropout(
                context_out, p=self.dropout_level, training=self.training)
            # (batch x emb_dim x seq_len)
            context_out = context_out.transpose(1, 2)
            context_out = F.relu(context_out)
            # (batch x nb_dense_dims)
            context_out = self.context_encoder(context_out)
            context_out = F.dropout(
                context_out, p=self.dropout_level, training=self.training)
            context_out = F.relu(context_out)
            joined.append(context_out)

        joined = torch.cat(joined, 1)

        out = []
        if self.include_lemma:
            # maybe get lemma_out, (if training)
            lemma_out = (train_out or {}).get('lemma_out', None)
            out.append(self.lemma_decoder(
                token_out, context_out, token_context, lemma_out))

        if self.include_pos:
            out.append(self.pos_decoder(joined))

        if self.include_morph:
            out.append(self.morph_decoder(joined))

        return out

    def _wrapped_forward(self, train_in, train_out):
        out = self(train_in, train_out)
        out_dict = {}
        if self.include_lemma:
            out_dict['lemma_out'] = out.pop(0)

        if self.include_pos:
            out_dict['pos_out'] = out.pop(0)

        if self.include_morph:
            out_dict['morph_out'] = out.pop(0)

        return out_dict

    def loss(self, output, target, output_label):
        if output_label == 'lemma_out':

            if self.include_lemma == 'generate':
                # collapse batch and seq dimensions
                output = output.transpose(0, 1) \
                               .contiguous() \
                               .view(-1, output.size(-1))
                # remove bos from target
                target = target[:, 1:].contiguous()
                target = target.view(-1)
                return self.lemma_loss(output, target)

            else:
                return self.lemma_loss(output, target)

        elif output_label == 'pos_out':
            return self.pos_loss(output, target)

        elif output_label == 'morph_out':
            return self.morph_loss(output, target)

    def epoch(self, train_in, train_out):
        self.move_to_gpu(gpu=self.gpu)  # eventually move to gpu
        self.train()
        batches = BatchIterator.from_numpy(
            self.batch_size, train_in, trg=train_out, dev=False, gpu=self.gpu)
        progbar = Progbar(target=len(batches) * self.batch_size)
        epoch_losses = defaultdict(float)

        for batch in range(len(batches)):
            src, trg = batches[batch]
            log_losses = []
            self.optimizer.zero_grad()

            for out_label, out in self._wrapped_forward(src, trg).items():
                # loss
                loss = self.loss(out, trg[out_label], out_label)
                loss.backward(retain_graph=True)
                # report
                epoch_losses[out_label] += loss.data[0]
                log_losses.append((out_label + '_loss', loss.data[0]))

            # optimize
            self.optimizer.step()

            # report
            log_losses = [('loss', sum(l for _, l in log_losses))] + log_losses
            progbar.update((batch + 1) * self.batch_size, tuple(log_losses))

        return epoch_losses

    def predict(self, input_data, batch_size=None):
        self.move_to_gpu(gpu=self.gpu)  # eventually move to gpu
        self.eval()
        out = {}
        batches = BatchIterator.from_numpy(
            batch_size or self.batch_size, input_data, dev=True, gpu=self.gpu)

        for batch in range(len(batches)):
            src, trg = batches[batch]
            pred = self._wrapped_forward(src, trg)

            for output_label, output in pred.items():
                # unwrap output
                if isinstance(output, Variable):
                    output = output.data

                # compute batched predictions
                if output_label == 'lemma_out':
                    if self.include_lemma == 'generate':
                        # (seqlen x batch x vocab) -> (batch x seqlen x vocab)
                        array = output.transpose(0, 1).cpu().numpy()
                    else:           # 'label'
                        array = output.cpu().numpy()

                elif output_label == 'pos_out':
                    array = output.cpu().numpy()

                else:               # 'morph_out'
                    array = output.cpu().numpy()

                # concatenate to previous batch predictions
                if output_label not in out:
                    out[output_label] = array

                else:
                    array = np.concatenate([out[output_label], array])
                    out[output_label] = array

        return out

    @staticmethod
    def load(model_dir, **kwargs):
        with open(os.path.join(model_dir, 'model_architecture.pt'), 'rb') as f:
            return torch.load(f)

    def save(self, model_dir):
        self.cpu()
        with open(os.path.join(model_dir, 'model_architecture.pt'), 'wb') as f:
            torch.save(self, f)
