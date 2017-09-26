from unittest import TestCase
import os
import shutil

from pandora.tagger import Tagger


class TestConfigLoader(TestCase):
    def setUp(self):
        try:
            os.makedirs("fake_model")
        except:
            pass

    def tearDown(self):
        shutil.rmtree("fake_model")

    def test_load(self):
        """ Ensure params are correctly loaded """
        tagger = Tagger(config_path="./tests/test_configs/config_chrestien.txt")
        self.assertEqual(tagger.nb_encoding_layers, 2, "nb_encoding_layers should be correctly loaded")
        self.assertEqual(tagger.nb_epochs, 3, "nb_epochs should be correctly loaded")
        self.assertEqual(tagger.nb_dense_dims, 1000, "nb_dense_dims should be correctly loaded")
        self.assertEqual(tagger.batch_size, 100, "batch_size should be correctly loaded")
        self.assertEqual(tagger.nb_left_tokens, 2, "nb_left_tokens should be correctly loaded")
        self.assertEqual(tagger.nb_right_tokens, 1, "nb_right_tokens should be correctly loaded")
        self.assertEqual(tagger.nb_context_tokens, 3, "nb_context_tokens should be correctly computed")
        self.assertEqual(tagger.nb_embedding_dims, 100, "nb_embedding_dims should be correctly loaded")
        self.assertEqual(tagger.model_dir, "fake_model", "model_dir should be correctly loaded")
        self.assertEqual(tagger.postcorrect, False, "postcorrect should be correctly loaded")
        self.assertEqual(tagger.nb_filters, 100, "nb_filters should be correctly loaded")
        self.assertEqual(tagger.filter_length, 3, "filter_length should be correctly loaded")
        self.assertEqual(tagger.focus_repr, "convolutions", "focus_repr should be correctly loaded")
        self.assertEqual(tagger.dropout_level, 0.15, "dropout_level should be correctly loaded")
        self.assertEqual(tagger.include_token, True, "include_token should be correctly loaded")
        self.assertEqual(tagger.include_context, True, "include_context should be correctly loaded")
        self.assertEqual(tagger.include_lemma, "label", "include_lemma should be correctly loaded")
        self.assertEqual(tagger.include_pos, True, "include_pos should be correctly loaded")
        self.assertEqual(tagger.include_morph, False, "include_morph should be correctly loaded")
        self.assertEqual(tagger.include_dev, True, "include_dev should be correctly loaded")
        self.assertEqual(tagger.include_test, True, "include_test should be correctly loaded")
        self.assertEqual(tagger.min_token_freq_emb, 5, "min_token_freq_emb should be correctly loaded")
        self.assertEqual(tagger.halve_lr_at, 75, "halve_lr_at should be correctly loaded")
        self.assertEqual(tagger.max_token_len, 20, "max_token_len should be correctly loaded")
        self.assertEqual(tagger.min_lem_cnt, 1, "min_lem_cnt should be correctly loaded")

    def test_load_after_save(self):
        """ Ensure param are correctly saved """
        tagger = Tagger(config_path="./tests/test_configs/config_chrestien.txt")
        tagger.include_pos = False
        tagger.curr_nb_epochs = 10
        tagger.save_params()
        del tagger

        tagger = Tagger(config_path="./fake_model/config.txt")
        self.assertEqual(tagger.nb_encoding_layers, 2, "nb_encoding_layers should be correctly loaded")
        self.assertEqual(tagger.nb_epochs, 3, "nb_epochs should be correctly loaded")
        self.assertEqual(tagger.nb_dense_dims, 1000, "nb_dense_dims should be correctly loaded")
        self.assertEqual(tagger.batch_size, 100, "batch_size should be correctly loaded")
        self.assertEqual(tagger.nb_left_tokens, 2, "nb_left_tokens should be correctly loaded")
        self.assertEqual(tagger.nb_right_tokens, 1, "nb_right_tokens should be correctly loaded")
        self.assertEqual(tagger.nb_context_tokens, 3, "nb_context_tokens should be correctly computed")
        self.assertEqual(tagger.nb_embedding_dims, 100, "nb_embedding_dims should be correctly loaded")
        self.assertEqual(tagger.model_dir, "fake_model", "model_dir should be correctly loaded")
        self.assertEqual(tagger.postcorrect, False, "postcorrect should be correctly loaded")
        self.assertEqual(tagger.nb_filters, 100, "nb_filters should be correctly loaded")
        self.assertEqual(tagger.filter_length, 3, "filter_length should be correctly loaded")
        self.assertEqual(tagger.focus_repr, "convolutions", "focus_repr should be correctly loaded")
        self.assertEqual(tagger.dropout_level, 0.15, "dropout_level should be correctly loaded")
        self.assertEqual(tagger.include_token, True, "include_token should be correctly loaded")
        self.assertEqual(tagger.include_context, True, "include_context should be correctly loaded")
        self.assertEqual(tagger.include_lemma, "label", "include_lemma should be correctly loaded")
        self.assertEqual(tagger.include_pos, False, "include_pos should be correctly loaded")
        self.assertEqual(tagger.include_morph, False, "include_morph should be correctly loaded")
        self.assertEqual(tagger.include_dev, True, "include_dev should be correctly loaded")
        self.assertEqual(tagger.include_test, True, "include_test should be correctly loaded")
        self.assertEqual(tagger.min_token_freq_emb, 5, "min_token_freq_emb should be correctly loaded")
        self.assertEqual(tagger.halve_lr_at, 75, "halve_lr_at should be correctly loaded")
        self.assertEqual(tagger.max_token_len, 20, "max_token_len should be correctly loaded")
        self.assertEqual(tagger.min_lem_cnt, 1, "min_lem_cnt should be correctly loaded")
        self.assertEqual(tagger.curr_nb_epochs, 10, "Current number of epochs should be correctly loaded")