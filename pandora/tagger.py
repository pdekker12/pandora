#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import json
import shutil

import numpy as np

import editdistance

import pandora.utils as utils
import pandora.evaluation as evaluation
from pandora.impl import KerasModel, PyTorchModel
from pandora.preprocessing import Preprocessor
from pandora.pretraining import Pretrainer
from pandora.logger import Logger


MODELS = {KerasModel.CONFIG_KEY: KerasModel, PyTorchModel.CONFIG_KEY: PyTorchModel}


class Tagger():
    def __init__(self, settings, config_path=None, overwrite=None, load=False):

        # initialize:
        self.setup = False
        self.settings = settings
        self.curr_nb_epochs = 0

        self.train_tokens, self.dev_tokens, self.test_tokens = None, None, None
        self.train_lemmas, self.dev_lemmas, self.test_lemmas = None, None, None
        self.train_pos, self.dev_pos, self.test_pos = None, None, None
        self.train_morph, self.dev_morph, self.test_morph = None, None, None
        self.embed_tokens = None

        self.logger = Logger()  # Default logger uses shell

        #self.test_batch_size = test_batch_size or batch_size

        if MODELS[settings.backend] is None:
            raise ValueError("Couldn't load implementation {}".format(settings.backend))

        if config_path is not None:
            self.settings = utils.parse_param_file(config_path)
        elif load is True:
            if settings.model_dir:
                config_path = os.sep.join((settings.model_dir, 'config.txt'))
                self.settings = utils.parse_param_file(config_path)
                print('Using params from config file: %s' % config_path)
            else:
                raise ValueError(
                    'To load a tagger you, must specify a model_name!')
        else:
            self.settings = settings

        if overwrite is not None:
            # Overwrite should be a dict of attributes to update the tagger
            for k, v in overwrite.items():
                self.settings[k] = v

        # create a models directory if it isn't there already
        if not os.path.isdir(settings.model_dir):
            os.mkdir(settings.model_dir)

        if load:
            self.load()

    @staticmethod
    def setup_from_disk(
            config_path,
            train_data=None, dev_data=None, test_data=None, verbose=False,
            load=False, **kwargs
    ):
        """ Command to load the whole tagger through a config.txt file and the path to some data

        :param config_path:
        :param train_data:
        :param dev_data:
        :param test_data:
        :return:
        :rtype: Tagger
        """
        if verbose:
            print('::: started :::')

        settings = utils.parse_param_file(config_path, verbose)

        train_data = utils.load_annotated_dir(
            directory=train_data,
            format='tab',
            extension='.tab',
            include_pos=settings.include_pos,
            include_lemma=settings.include_lemma,
            include_morph=settings.include_morph,
            nb_instances=None
        )

        if not len(train_data.keys()) or \
                not len(train_data[list(train_data.keys())[0]]):
            raise ValueError('No training data loaded...')

        data_sets = dict(
            train_data=train_data,
        )

        if dev_data:
            dev_data = utils.load_annotated_dir(
                directory=dev_data,
                format='tab',
                extension='.tab',
                include_pos=settings.include_pos,
                include_lemma=settings.include_lemma,
                include_morph=settings.include_morph,
                nb_instances=None
            )
            if not len(dev_data.keys()) or \
                    not len(dev_data[list(dev_data.keys())[0]]):
                raise ValueError('No dev data loaded...')
            data_sets["dev_data"] = dev_data

        if test_data:
            test_data = utils.load_annotated_dir(
                directory=test_data,
                format='tab',
                extension='.tab',
                include_pos=settings.include_pos,
                include_lemma=settings.include_lemma,
                include_morph=settings.include_morph,
                nb_instances=None
            )
            if not len(test_data.keys()) or \
                    not len(test_data[list(test_data.keys())[0]]):
                raise ValueError('No test data loaded...')
            data_sets["test_data"] = test_data

        if settings.embed:
            embed_data = utils.load_annotated_dir(
                settings.embed,
                format='tab',
                extension='.tab',
                include_pos=False,
                include_lemma=False,
                include_morph=False,
                nb_instances=None
            )
            if not len(embed_data.keys()) or \
                    not len(embed_data[list(embed_data.keys())[0]]):
                raise ValueError('No embeddings data loaded...')
            data_sets["embed_data"] = embed_data

        if load:
            if verbose:
                print('::: loading model :::')
            tagger = Tagger(load=True, settings=settings)
            if tagger.config_path == os.sep.join((tagger.model_dir, 'config.txt')):
                shutil.copy(tagger.settings.config_path, os.sep.join((tagger.settings.model_dir, 'config_original.txt')))
                if verbose:
                    print('Warning: current config file will be overwritten. Saving it to config_original.txt')
            tagger.setup_to_train(build=False, **data_sets)
            tagger.curr_nb_epochs = int(params['curr_nb_epochs'])
            if verbose:
                print("restart from epoch " + str(tagger.curr_nb_epochs) + "...")
            tagger.setup = True
        else:
            tagger = Tagger(settings=settings)
            tagger.setup_to_train(**data_sets)

        return tagger

    def load(self):
        print('Re-loading preprocessor...')
        self.preprocessor = Preprocessor(categorical=self.backend == 'Keras')
        self.preprocessor.load(model_dir=self.model_dir,
                               max_token_len=self.max_token_len,
                               max_lemma_len=self.max_lemma_len,
                               focus_repr=self.focus_repr,
                               include_lemma=self.include_lemma,
                               include_pos=self.include_pos,
                               include_morph=self.include_morph)

        print('Re-loading pretrainer...')
        self.pretrainer = Pretrainer()
        self.pretrainer.load(model_dir=self.model_dir,
                             nb_left_tokens=self.nb_left_tokens,
                             nb_right_tokens=self.nb_right_tokens)

        print('Re-building model...')
        self.model = MODELS[self.model].load(
            self.model_dir,
            include_lemma=self.include_lemma,
            include_pos=self.include_pos,
            include_morph=self.include_morph)

        if self.include_lemma:
            print('Loading known lemmas...')
            lemmas_path = os.sep.join((self.model_dir, 'known_lemmas.txt'))
            self.known_lemmas = set([l.strip() for l in open(lemmas_path, 'r')])

    def setup_to_train(self, train_data, dev_data=None, test_data=None,
                       embed_data=None, build=True):
        if build:
            # create a model directory:
            if os.path.isdir(self.settings.model_dir):
                shutil.rmtree(self.settings.model_dir)
            os.mkdir(self.settings.model_dir)

        self.train_tokens = train_data['token']
        if self.settings.include_test:
            self.test_tokens = test_data['token']
        if self.settings.include_dev:
            self.dev_tokens = dev_data['token']

        if self.settings.include_lemma:
            self.train_lemmas = train_data['lemma']
            self.known_lemmas = set(self.train_lemmas)
            if self.settings.include_dev:
                self.dev_lemmas = dev_data['lemma']
            if self.settings.include_test:
                self.test_lemmas = test_data['lemma']

        if self.settings.include_pos:
            self.train_pos = train_data['pos']
            if self.settings.include_dev:
                self.dev_pos = dev_data['pos']
            if self.settings.include_test:
                self.test_pos = test_data['pos']

        if self.settings.include_morph:
            self.settings.train_morph = train_data['morph']
            if self.settings.include_dev:
                self.dev_morph = dev_data['morph']
            if self.settings.include_test:
                self.test_morph = test_data['morph']

        if embed_data:
            self.embed_tokens = embed_data['token']
            #TODO: concatenate with train tokens ?
            #in that case:
            #self.embed_tokens = dict(embed_data['tokens'].items() + train_data['token'].items() )

        self.preprocessor = Preprocessor().fit(
            tokens=self.train_tokens,
            lemmas=self.train_lemmas,
            pos=self.train_pos,
            morph=self.train_morph,
            settings=self.settings)
        self.settings = self.preprocessor.settings

        self.pretrainer = Pretrainer(self.settings)
        if self.embed_tokens is not None:
            self.pretrainer.fit(tokens=self.embed_tokens)
        else:
            self.pretrainer.fit(tokens=self.train_tokens)
        self.settings = self.pretrainer.settings

        train_transformed = self.preprocessor.transform(
            tokens=self.train_tokens,
            lemmas=self.train_lemmas,
            pos=self.train_pos,
            morph=self.train_morph)

        if self.settings.include_dev:
            dev_transformed = self.preprocessor.transform(
                tokens=self.dev_tokens,
                lemmas=self.dev_lemmas,
                pos=self.dev_pos,
                morph=self.dev_morph)

        if self.settings.include_test:
            test_transformed = self.preprocessor.transform(
                tokens=self.test_tokens,
                lemmas=self.test_lemmas,
                pos=self.test_pos,
                morph=self.test_morph)

        self.train_X_focus = train_transformed['X_focus']
        if self.settings.include_dev:
            self.dev_X_focus = dev_transformed['X_focus']
        if self.settings.include_test:
            self.test_X_focus = test_transformed['X_focus']

        if self.settings.include_lemma:
            self.train_X_lemma = train_transformed['X_lemma']
            if self.settings.include_dev:
                self.dev_X_lemma = dev_transformed['X_lemma']
            if self.settings.include_test:
                self.test_X_lemma = test_transformed['X_lemma']

        if self.settings.include_pos:
            self.train_X_pos = train_transformed['X_pos']
            if self.settings.include_dev:
                self.dev_X_pos = dev_transformed['X_pos']
            if self.settings.include_test:
                self.test_X_pos = test_transformed['X_pos']

        if self.settings.include_morph:
            self.train_X_morph = train_transformed['X_morph']
            if self.settings.include_dev:
                self.dev_X_morph = dev_transformed['X_morph']
            if self.settings.include_test:
                self.test_X_morph = test_transformed['X_morph']

        self.train_contexts = self.pretrainer.transform(
            tokens=self.train_tokens)
        if self.settings.include_dev:
            self.dev_contexts = self.pretrainer.transform(
                tokens=self.dev_tokens)
        if self.settings.include_test:
            self.test_contexts = self.pretrainer.transform(
                tokens=self.test_tokens)

        # remove unused items from memory:
        try:
            del train_transformed
        except:
            pass
        try:
            del dev_transformed
        except:
            pass
        try:
            del test_transformed
        except:
            pass

        print('Building model...')
        self.model = MODELS[self.settings.backend](settings=self.settings)
        self.model.print_summary()
        self.settings = self.model.settings

        self.setup = True
        self.save()

    def train(self, nb_epochs=None):
        if nb_epochs:
            self.settings.nb_epochs = nb_epochs
        for i in range(self.settings.nb_epochs):
            scores = self.epoch()
        return scores

    def print_stats(self):
        print('Train stats:')
        utils.stats(tokens=self.train_tokens,
                    lemmas=self.train_lemmas,
                    known=self.preprocessor.known_tokens)
        print('Test stats:')
        utils.stats(tokens=self.test_tokens,
                    lemmas=self.test_lemmas,
                    known=self.preprocessor.known_tokens)

    def test(self, multilabel_threshold=0.5):
        """ Run tests evaluation with prediction

        :param multilabel_threshold:
        :return: Score Dictionary
        """
        if not self.include_test:
            raise ValueError(
                'Please do not call .test() if no test data is available.')

        score_dict = {}

        # get test predictions:
        test_in = {}
        if self.settings.include_token:
            test_in['focus_in'] = self.test_X_focus
        if self.settings.include_context:
            test_in['context_in'] = self.test_contexts

        test_preds = self.model.predict(test_in, batch_size=self.test_batch_size)

        if isinstance(test_preds, np.ndarray):
            test_preds = [test_preds]

        if self.settings.include_lemma:
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(
                predictions=test_preds['lemma_out'])
            if self.settings.postcorrect:
                for i in range(len(pred_lemmas)):
                    if pred_lemmas[i] not in self.known_lemmas:
                        pred_lemmas[i] = min(
                            self.known_lemmas,
                            key=lambda x: editdistance.eval(x, pred_lemmas[i]))

            score_dict['test_lemma'] = evaluation.single_label_accuracies(
                gold=self.test_lemmas,
                silver=pred_lemmas,
                test_tokens=self.test_tokens,
                known_tokens=self.preprocessor.known_tokens,
                print_scores=False)

        if self.settings.include_pos:

            pred_pos = self.preprocessor.inverse_transform_pos(
                predictions=test_preds['pos_out'])
            score_dict['test_pos'] = evaluation.single_label_accuracies(
                gold=self.test_pos,
                silver=pred_pos,
                test_tokens=self.test_tokens,
                known_tokens=self.preprocessor.known_tokens,
                print_scores=False)

        if self.settings.include_morph:
            pred_morph = self.preprocessor.inverse_transform_morph(
                predictions=test_preds['morph_out'],
                threshold=multilabel_threshold)
            if self.include_morph == 'label':
                score_dict['test_morph'] = evaluation.single_label_accuracies(
                    gold=self.test_morph,
                    silver=pred_morph,
                    test_tokens=self.test_tokens,
                    known_tokens=self.preprocessor.known_tokens,
                    print_scores=False)
            elif self.include_morph == 'multilabel':
                score_dict['test_morph'] = evaluation.multilabel_accuracies(
                    gold=self.test_morph,
                    silver=pred_morph,
                    test_tokens=self.test_tokens,
                    known_tokens=self.preprocessor.known_tokens,
                    print_scores=False)

        return score_dict

    def save(self):
        # save architecture:
        self.model.save(self.settings.model_dir)

        # save preprocessor:
        self.preprocessor.save(self.settings.model_dir)
        self.pretrainer.save(self.settings.model_dir)

        if self.settings.include_lemma:
            lemmas_path = os.sep.join((self.settings.model_dir, 'known_lemmas.txt'))
            with open(lemmas_path, 'w') as f:
                f.write('\n'.join(sorted(self.known_lemmas)))

        self.save_params()

    def save_params(self):
        """ Write the params to self.model_dir """
        with open(os.sep.join((self.settings.model_dir, 'config.txt')), 'w') as f:
            if hasattr(self, 'curr_nb_epochs'):
                self.settings.curr_nb_epochs = self.curr_nb_epochs
            f.write(json.dumps(self.settings))

    def epoch(self, autosave=True, eval_test=False):
        if not self.setup:
            raise ValueError(
                'Not set up yet... Call Tagger.setup_to_train() first.')

        # update nb of epochs ran so far:
        self.curr_nb_epochs += 1
        print("-> Epoch ", self.curr_nb_epochs, "...")

        if self.curr_nb_epochs and self.settings.halve_lr_at:
            # update learning rate at specific points:
            if self.curr_nb_epochs % self.settings.halve_lr_at == 0:
                self.model.adjust_lr()

        # get inputs and outputs straight:
        train_in, train_out = {}, {}
        if self.settings.include_token:
            train_in['focus_in'] = self.train_X_focus
        if self.settings.include_context:
            train_in['context_in'] = self.train_contexts
        if self.settings.include_lemma:
            train_out['lemma_out'] = self.train_X_lemma
        if self.settings.include_pos:
            train_out['pos_out'] = self.train_X_pos
        if self.settings.include_morph:
            train_out['morph_out'] = self.train_X_morph

        self.model.epoch(train_in, train_out)

        def run_eval():
            score_dict = self.eval(train_in=train_in)
            if eval_test is True:
                score_dict.update(self.test())
            return score_dict

        score_dict = self.logger.epoch(self.curr_nb_epochs, run_eval)

        if autosave:
            self.save()

        return score_dict

    def eval(self, train_in):
        """ Evaluate current epoch

        :param train_in: Data from training
        :return:
        """

        # get train preds:
        train_preds = self.model.predict(train_in, batch_size=self.test_batch_size)
        if isinstance(train_preds, np.ndarray):
            train_preds = [train_preds]

        # get dev preds:
        dev_preds = None
        if self.include_dev:
            dev_in = {}
            if self.include_token:
                dev_in['focus_in'] = self.dev_X_focus
            if self.include_context:
                dev_in['context_in'] = self.dev_contexts

            dev_preds = self.model.predict(dev_in, batch_size=self.test_batch_size)

            if isinstance(dev_preds, np.ndarray):
                dev_preds = [dev_preds]

        return self.get_score_dict(train_preds, dev_preds)

    def get_score_dict(self, train_preds, dev_preds=None):
        """ Get score dict given train and dev preds

        :param train_preds: Training predictions
        :param dev_preds: Development predictions
        :return: Score dictionary
        """

        score_dict = {}
        if self.include_lemma:
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(
                predictions=train_preds['lemma_out'])
            score_dict['train_lemma'] = evaluation.single_label_accuracies(
                gold=self.train_lemmas,
                silver=pred_lemmas,
                test_tokens=self.train_tokens,
                known_tokens=self.preprocessor.known_tokens,
                print_scores=False)

            if self.include_dev:
                pred_lemmas = self.preprocessor.inverse_transform_lemmas(
                    predictions=dev_preds['lemma_out'])
                score_dict['dev_lemma'] = evaluation.single_label_accuracies(
                    gold=self.dev_lemmas,
                    silver=pred_lemmas,
                    test_tokens=self.dev_tokens,
                    known_tokens=self.preprocessor.known_tokens,
                    print_scores=False)

                if self.postcorrect:
                    for i in range(len(pred_lemmas)):
                        if pred_lemmas[i] not in self.known_lemmas:
                            pred_lemmas[i] = min(
                                self.known_lemmas,
                                key=lambda x: editdistance.eval(x, pred_lemmas[i]))

                    score_dict['dev_lemma_postcorrect'] = evaluation.single_label_accuracies(
                        gold=self.dev_lemmas,
                        silver=pred_lemmas,
                        test_tokens=self.dev_tokens,
                        known_tokens=self.preprocessor.known_tokens)

        if self.include_pos:

            pred_pos = self.preprocessor.inverse_transform_pos(
                predictions=train_preds['pos_out'])
            score_dict['train_pos'] = evaluation.single_label_accuracies(
                gold=self.train_pos,
                silver=pred_pos,
                test_tokens=self.train_tokens,
                known_tokens=self.preprocessor.known_tokens,
                print_scores=False)

            if self.include_dev:
                pred_pos = self.preprocessor.inverse_transform_pos(
                    predictions=dev_preds['pos_out'])
                score_dict['dev_pos'] = evaluation.single_label_accuracies(
                    gold=self.dev_pos,
                    silver=pred_pos,
                    test_tokens=self.dev_tokens,
                    known_tokens=self.preprocessor.known_tokens,
                    print_scores=False)

        if self.include_morph:

            pred_morph = self.preprocessor.inverse_transform_morph(
                predictions=train_preds['morph_out'])

            if self.include_morph == 'label':
                score_dict['train_morph'] = evaluation.single_label_accuracies(
                    gold=self.train_morph,
                    silver=pred_morph,
                    test_tokens=self.train_tokens,
                    known_tokens=self.preprocessor.known_tokens,
                    print_scores=False)

            elif self.include_morph == 'multilabel':
                score_dict['train_morph'] = evaluation.multilabel_accuracies(
                    gold=self.train_morph,
                    silver=pred_morph,
                    test_tokens=self.train_tokens,
                    known_tokens=self.preprocessor.known_tokens,
                    print_scores=False)

            if self.include_dev:

                pred_morph = self.preprocessor.inverse_transform_morph(
                    predictions=dev_preds['morph_out'])

                if self.include_morph == 'label':
                    score_dict['dev_morph'] = evaluation.single_label_accuracies(
                        gold=self.train_morph,
                        silver=pred_morph,
                        test_tokens=self.dev_tokens,
                        known_tokens=self.preprocessor.known_tokens,
                        print_scores=False)

                elif self.include_morph == 'multilabel':
                    score_dict['dev_morph'] = evaluation.multilabel_accuracies(
                        gold=self.train_morph,
                        silver=pred_morph,
                        test_tokens=self.dev_tokens,
                        known_tokens=self.preprocessor.known_tokens,
                        print_scores=False)

        return score_dict

    def annotate(self, tokens):
        X_focus = self.preprocessor.transform(tokens=tokens)['X_focus']
        X_context = self.pretrainer.transform(tokens=tokens)

        new_in = {}
        if self.settings.include_token:
            new_in['focus_in'] = X_focus
        if self.settings.include_context:
            new_in['context_in'] = X_context

        preds = self.model.predict(new_in)
        if isinstance(preds, np.ndarray):
            preds = [preds]

        annotation_dict = {'tokens': tokens}
        if self.settings.include_lemma:
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(
                predictions=preds['lemma_out'])
            annotation_dict['lemmas'] = pred_lemmas
            if self.settings.postcorrect:
                for i in range(len(pred_lemmas)):
                    if pred_lemmas[i] not in self.known_lemmas:
                        pred_lemmas[i] = min(
                            self.known_lemmas,
                            key=lambda x: editdistance.eval(x, pred_lemmas[i]))
                annotation_dict['postcorrect_lemmas'] = pred_lemmas

        if self.settings.include_pos:
            pred_pos = self.preprocessor.inverse_transform_pos(
                predictions=preds['pos_out'])
            annotation_dict['pos'] = pred_pos

        if self.settings.include_morph:
            pred_morph = self.preprocessor.inverse_transform_morph(
                predictions=preds['morph_out'])
            annotation_dict['morph'] = pred_morph

        return annotation_dict
