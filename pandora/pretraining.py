#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

from collections import Counter
from operator import itemgetter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE


class SentenceIterator:
    """
    Gensim-style iterator to loop over the training
    data in batches.
    """
    def __init__(self, tokens, sentence_len=100):
        """
        Constructor.

        Parameters
        ===========
        directory : list of str
            Tokens over which to iterate.
        sentence_len : int (default: 100)
            Batch length
        """

        self.sentence_len = sentence_len
        self.tokens = [t.lower() for t in tokens]
        self.idxs = []
        start_idx, end_idx = 0, self.sentence_len
        while end_idx < len(self.tokens):
            self.idxs.append((start_idx, end_idx))
            start_idx += self.sentence_len
            end_idx += self.sentence_len

    def __iter__(self):
        """
        Simple batch iterator.
        """
        for start_idx, end_idx in self.idxs:
            yield self.tokens[start_idx: end_idx]


class Pretrainer:
    """
    Tagger component which:
    * pretrains word embeddings
      for the context representation of tokens. These
      weights can be used to initialize the word
      embedding weights of the tagger.
    * can be used to shingle a list of tokens into
      the context vectors for the model.
    """

    def __init__(self, nb_left_tokens=None, nb_right_tokens=None,
                 sentence_len=100, window=5,
                 minimum_count=2, size=300, nb_mfi=500,
                 nb_workers=10, nb_negative=5):
        """
        Constructor.

        Parameters
        ===========
        nb_left_tokens : int
            Number of tokens left of the focus token to consider
        nb_right_tokens : int
            Number of tokens right of the focus token to consider
        sentence_len : int (default = 100)
            Batch length for the embeddings iterator.
        window : int (default = 5)
            Size of the window for the embeddings model (Gensim)
        minimum_count : int (default = 2)
            Minimum frequency of a token to be included in the
            embeddings model. Tokens with a frequency < minimum_count
            are mapped to the dummy '<UNK>' token.
        size : int (default = 300)
            Dimensionality of the Gensim model.
        nb_mfi : int (default = 500)
            Number of most frequent tokens for viz purposes.
        nb_workers : int (default = 10)
            Nb of workers (for the Gensim model).
        nb_negative : int (default = 5)
            Nb of negative samples to be drawn by the Gensim model.

        """
        self.nb_left_tokens = nb_left_tokens
        self.nb_right_tokens = nb_right_tokens
        self.size = size
        self.nb_mfi = nb_mfi
        self.window = window
        self.minimum_count = minimum_count
        self.nb_workers = nb_workers
        self.nb_negative = nb_negative

    def fit(self, tokens, viz=False):
        """
        Fit the pretrainer on a list of (training) tokens.

        Parameters
        ===========
        tokens : list of str
            List of training tokens on which to pretrain
        viz : bool (default = False)
            Whether to visualize the embeddings using a vanilla TSNE.

        """
        # get most frequent items for plotting:
        tokens = [t.lower() for t in tokens]
        self.mfi = [t for t, _ in Counter(tokens).most_common(self.nb_mfi)]

        self.sentence_iterator = SentenceIterator(tokens=tokens)
        # train embeddings:
        self.w2v_model = Word2Vec(
            self.sentence_iterator,
            window=self.window,
            min_count=self.minimum_count,
            size=self.size,
            workers=self.nb_workers,
            negative=self.nb_negative)

        if viz:
            self.plot_mfi()
            self.most_similar()

        # build an index of the train tokens
        # which occur at least minimum_count times:
        self.token_idx = {'<UNK>': 0}
        for k, v in Counter(tokens).items():
            if v >= self.minimum_count:
                self.token_idx[k] = len(self.token_idx)

        # create an ordered vocab:
        self.train_token_vocab = [k for k, v in sorted(self.token_idx.items(),
                                                       key=itemgetter(1))]
        self.pretrained_embeddings = self.get_weights(self.train_token_vocab)

        return self

    def get_weights(self, vocab):
        """
        Get the weight matrix for a list of tokens.

        Parameters
        ===========
        vocab : iterable of str
            List of tokens for which to return weights.

        Returns
        ===========
        An embedding numpy matrix (np.float32) of shape
        (len(vocab), dimensionality).

        """
        unk = np.zeros(self.size)
        weights = []
        for w in vocab:
            try:
                weights.append(self.w2v_model[w])
            except KeyError:
                weights.append(unk)
        return [np.asarray(weights, dtype='float32')]

    def transform(self, tokens):
        """
        Transform a list of tokens to context vectors.

        Parameters
        ===========
        tokens : iterable of str
            List of tokens for which to create context vectors.

        Returns
        ===========
        An integer numpy matrix (np.int32) of shape
        (nb_tokens, nb_left_tokens + nb_right_tokens).

        """
        context_ints = []
        tokens = [t.lower() for t in tokens]
        for curr_idx, token in enumerate(tokens):
            ints = []
            # vectorize left context:
            left_context_tokens = [tokens[curr_idx-(t+1)]
                                   for t in range(self.nb_left_tokens)
                                   if curr_idx-(t+1) >= 0][::-1]

            idxs = []
            if left_context_tokens:
                idxs = [self.token_idx[t] if t in self.token_idx else 0
                        for t in left_context_tokens]
            while len(idxs) < self.nb_left_tokens:
                idxs = [0] + idxs
            ints.extend(idxs)

            # vectorize right context
            right_context_tokens = [tokens[curr_idx+(t+1)]
                                    for t in range(self.nb_right_tokens)
                                    if curr_idx+(t+1) < len(tokens)]

            idxs = []
            if right_context_tokens:
                idxs = [self.token_idx[t] if t in self.token_idx else 0
                        for t in right_context_tokens]
            while len(idxs) < self.nb_right_tokens:
                idxs.append(0)
            ints.extend(idxs)

            context_ints.append(ints)

        return np.asarray(context_ints, dtype='int32')

    def plot_mfi(self, outputfile='embeddings.pdf', nb_clusters=8):
        """
        Plots the current state of the embeddings model using a vanilla TSNE.

        Parameters
        ===========
        outputfile : str (default = embeddings.pdf)
            Path to the file where the model should be saved.
        nb_clusters : str (default = 8)
            Number of clusters to color as a reading aid.
        """
        X = np.asarray([self.w2v_model[w] for w in self.mfi
                        if w in self.w2v_model], dtype='float32')
        tsne = TSNE(n_components=2)
        coor = tsne.fit_transform(X)

        plt.clf()
        sns.set_style('dark')
        sns.plt.rcParams['axes.linewidth'] = 0.4
        fig, ax1 = sns.plt.subplots()

        labels = self.mfi
        # first plot slices:
        x1, x2 = coor[:, 0], coor[:, 1]
        ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none')
        # clustering on top (add some colouring):
        clustering = AgglomerativeClustering(
            linkage='ward',
            affinity='euclidean', n_clusters=nb_clusters)
        clustering.fit(coor)
        # add names:
        for x, y, name, cluster_label in zip(x1, x2, labels,
                                             clustering.labels_):
            ax1.text(x, y, name, ha='center', va="center",
                     color=plt.cm.spectral(cluster_label / 10.),
                     fontdict={'family': 'sans-serif', 'size': 8})
        # control aesthetics:
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_xticklabels([])
        ax1.set_xticks([])
        ax1.set_yticklabels([])
        ax1.set_yticks([])
        sns.plt.savefig(outputfile, bbox_inches=0)
        sns.plt.close()

    def save(self, model_dir):
        """
        Serializes the pretrainer through saving the indices
        to the embeddings vocabulary in plain text format.

        Parameters
        ===========
        model_dir : str
            Path to the model directory where the model should be saved.
        """
        with open(os.sep.join((model_dir, 'token_idx.txt')), 'w') as f:
            for tok, idx in self.token_idx.items():
                f.write('\t'.join((tok, str(idx)))+'\n')

    def load(self, model_dir, nb_left_tokens, nb_right_tokens):
        """
        Loads a previously fitted pretrainer.

        Parameters
        ===========
        model_dir : str
            Path to the model directory from where the model should
            be loaded.
        """
        self.nb_left_tokens = nb_left_tokens
        self.nb_right_tokens = nb_right_tokens

        self.token_idx = {}
        for line in open(os.sep.join((model_dir, 'token_idx.txt')), 'r'):
            tok, idx = line.strip().split()
            self.token_idx[tok] = int(idx)

        # create an ordered vocab:
        self.train_token_vocab = [k for k, v in sorted(
            self.token_idx.items(), key=itemgetter(1))]
