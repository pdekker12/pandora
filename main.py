#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import pandora.utils
from pandora.tagger import Tagger
import os
import shutil

def main(config, train, dev, test=None, load=False, verbose=True, **kwargs):
    """ Main CLI Interface

    :param config: Path to retrieve configuration file
    :type config: str
    :param train: Path to directory containing dev files
    :type train: str
    :param dev: Path to directory containing test files
    :type dev: str
    :param test: Path to directory containing train files
    :type test: str
    :param load: Whether to load an existing model to train on top of it (default: False)
    :type load: bool
    :param nb_epochs: Number of epoch
    :type nb_epochs: int
    :param kwargs: Other arguments
    :type kwargs: dict
    :return:
    """
    print('::: started :::')
    params = pandora.utils.get_param_dict(config)
    params['config_path'] = config
    params.update({k: v for k, v in kwargs.items() if v is not None})
    print("::: Loaded Config :::")
    for k, v in params.items():
        print("\t{} : {}".format(k, v))

    train_data = pandora.utils.load_annotated_dir(
        train,
        format='tab',
        extension='.tab',
        include_pos=params['include_pos'],
        include_lemma=params['include_lemma'],
        include_morph=params['include_morph'],
        nb_instances=None
    )

    dev_data = pandora.utils.load_annotated_dir(
        dev,
        format='tab',
        extension='.tab',
        include_pos=params['include_pos'],
        include_lemma=params['include_lemma'],
        include_morph=params['include_morph'],
        nb_instances=None
    )

    data_sets = dict(
            train_data=train_data,
            dev_data=dev_data
    )

    if test is not None:
        test_data = pandora.utils.load_annotated_dir(
            test,
            format='tab',
            extension='.tab',
            include_pos=params['include_pos'],
            include_lemma=params['include_lemma'],
            include_morph=params['include_morph'],
            nb_instances=None
        )
        data_sets["test_data"] = test_data

    if load:
        print('::: loading model :::')
        tagger = Tagger(load=True, model_dir=params['model_dir'])
        if tagger.config_path == os.sep.join((tagger.model_dir, 'config.txt')):
            shutil.copy(tagger.config_path, os.sep.join((tagger.model_dir, 'config_original.txt')))
            print('Warning: current config file will be overwritten. Saving it to config_original.txt')
        tagger.setup_to_train(build=False, **data_sets)
        tagger.curr_nb_epochs = int(params['curr_nb_epochs'])
        print("restart from epoch "+str(tagger.curr_nb_epochs)+"...")
        tagger.setup = True
    else:
        tagger = Tagger(**params)
        tagger.setup_to_train(**data_sets)

    nb_epochs = int(params['nb_epochs'])
    for i in range(nb_epochs):
        if i == 0 or i == nb_epochs - 1:
            curr_verbose = True
        else:
            curr_verbose = verbose
        tagger.epoch(verbose=curr_verbose, autosave=True)
        if test is not None:
            tagger.test(verbose=curr_verbose)

    tagger.save()
    print('::: ended :::')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training interface of Pandora")
    parser.add_argument("config", help="Path to retrieve configuration file")
    parser.add_argument("--dev", help="Path to directory containing dev files")
    parser.add_argument("--test", help="Path to directory containing test files")
    parser.add_argument("--train", help="Path to directory containing train files")
    parser.add_argument("--nb_epochs", help="Number of epoch", type=int)
    parser.add_argument(
        "--load",
        dest="load",
        action="store_true", 
        default=False,
        help="Whether to load an existing model to train on top of it (default: False)"
    )
    parser.add_argument(
        "--silent",
        dest="verbose",
        action="store_false",
        default=True,
        help="Stop printing results at each epoch"
    )

    main(**vars(parser.parse_args()))

