#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import Counter
from operator import itemgetter
import numpy as np

def single_label_accuracies(gold, silver,
                            test_tokens, known_tokens,
                            print_scores=True):
    """Calculate accuracy scores, assuming a single-label setup.
       Single-label accuracies are calculated for all tokens, as
       well as the unknown and known tokens separately. A known
       token is considered a token which has been encountered
       verbatimly during training.

    Parameters
    ===========
    gold : list of str
        The correct labels
    silver : list of str
        The predicted labels
    test_tokens : list of str
        The original tokens corresponding to
        the list of gold/silver labels.
    known_tokens : set of str
        A set of tokens which were met verbatimly
        during training (cf. known vs unknown tokens)
    print_scores : bool (default: True)
        Whether to print the scores (on top of
        returning them)

    Returns
    ===========
    results : tuple
        A 3-tuple of the:
            * overall accuracy
            * known tokens accuracy
            * unknown tokens accuracy

    """
    
    kno_corr, unk_corr = 0.0, 0.0
    nb_kno, nb_unk = 0.0, 0.0

    for gold_pred, silver_pred, tok in zip(gold, silver, test_tokens):

        if tok in known_tokens:
            nb_kno += 1
            if gold_pred == silver_pred:
                kno_corr += 1
        else:
            nb_unk += 1
            if gold_pred == silver_pred:
                unk_corr += 1

    all_acc = (kno_corr + unk_corr) / (nb_kno + nb_unk)
    kno_acc = kno_corr / nb_kno

    # account for situation with no unknowns:
    unk_acc = 0.0
    if nb_unk > 0:
        unk_acc = unk_corr / nb_unk

    if print_scores:
        print('+\tall acc:', all_acc)
        print('+\tkno acc:', kno_acc)
        print('+\tunk acc:', unk_acc)

    return all_acc, kno_acc, unk_acc

def multilabel_accuracies(gold, silver,
                          test_tokens, known_tokens,
                          print_scores=True):

    """Calculate accuracy scores, assuming a multi-label setup.
       Especially useful for calculating scores on complex morpho-
       logical tags. Multi-label accuracies are calculated for
       all tokens, as well as the unknown and known tokens
       separately. A known token is considered a token which has
       been encountered verbatimly during training.
       Note: sublabels are expected to be splitted using pipes.

    Parameters
    ===========
    gold : list of str
        The correct labels
    silver : list of str
        The predicted labels
    test_tokens : list of str
        The original tokens corresponding to
        the list of gold/silver labels.
    known_tokens : set of str
        A set of tokens which were met verbatimly
        during training (cf. known vs unknown tokens)
    print_scores : bool (default: True)
        Whether to print the scores (on top of
        returning them)

    Returns
    ===========
    results : tuple
        A 3-tuple of the:
            * overall accuracy
            * known tokens accuracy
            * unknown tokens accuracy

    """
    kno_corr, unk_corr = 0.0, 0.0
    nb_kno, nb_unk = 0.0, 0.0

    for gold_pred, silver_pred, tok in zip(gold, silver, test_tokens):
        gold_pred = set(gold_pred.split('|'))
        silver_pred = set(silver_pred.split('|'))
        if tok in known_tokens:
            nb_kno += 1
            if gold_pred == silver_pred:
                kno_corr += 1
        else:
            nb_unk += 1
            if gold_pred == silver_pred:
                unk_corr += 1

    all_acc = (kno_corr + unk_corr) / (nb_kno + nb_unk)
    kno_acc = kno_corr / nb_kno

    # account for situation with no unknowns:
    unk_acc = 0.0
    if nb_unk > 0:
        unk_acc = unk_corr / nb_unk

    if print_scores:
        print('+\tall acc:', all_acc)
        print('+\tkno acc:', kno_acc)
        print('+\tunk acc:', unk_acc)

    return all_acc, kno_acc, unk_acc


    