from unittest import TestCase
import mock
import os

from pandora.logger import Logger


class TestLogger(TestCase):
    def cleanUp(self):
        if os.path.isfile("test_logs.csv"):
            os.remove("test_logs.csv")

    def setUp(self):
        self.cleanUp()

    def tearDown(self):
        self.cleanUp()

    def test_print(self):
        """ Ensure print works normally """
        logger = Logger()
        logger.__print_function__ = mock.Mock()
        logger.write = mock.Mock()
        for i in range(100):
            logger.epoch(i, lambda: {"train_lemma": (0+i, 1+i, 2+i), "dev_lemma": (1+i, 2+i, 3+i)})

        self.assertEqual(logger.__print_function__.called, True, "Calling to print should have been done")
        self.assertEqual(logger.write.called, False, "File has not been set and should not be called")

        self.assertEqual(
            logger.__print_function__.call_args_list,
            [
                call
                for i in range(100)
                for call in [
                    mock.call("::: Train Scores (lemma) :::"),
                    mock.call('+\tall acc:', 0+i),
                    mock.call('+\tkno acc:', 1+i),
                    mock.call('+\tunk acc:', 2+i),
                    mock.call("::: Dev Scores (lemma) :::"),
                    mock.call('+\tall acc:', 1+i),
                    mock.call('+\tkno acc:', 2+i),
                    mock.call('+\tunk acc:', 3+i),
                ]
            ]
        )

    def test_print_each(self):
        """ Ensure print works normally """
        logger = Logger(each=3, nb_epochs=100)
        logger.__print_function__ = mock.Mock()
        logger.write = mock.Mock()
        for i in range(100):
            i += 1
            logger.epoch(i, lambda: {"train_lemma": (0+i, 1+i, 2+i)})

        self.assertEqual(logger.__print_function__.called, True, "Calling to print should have been done")
        self.assertEqual(logger.write.called, False, "File has not been set and should not be called")
        self.assertEqual(len(logger.__print_function__.call_args_list), 35*4, "There should be 35 time the printing")
        expected = [
            mock.call("::: Train Scores (lemma) :::"),
            mock.call('+\tall acc:', 1),
            mock.call('+\tkno acc:', 2),
            mock.call('+\tunk acc:', 3)
        ] + [
            call
            for i in range(1, 34)
            for call in [
                mock.call("::: Train Scores (lemma) :::"),
                mock.call('+\tall acc:', 0+i*3),
                mock.call('+\tkno acc:', 1+i*3),
                mock.call('+\tunk acc:', 2+i*3)
            ]
        ] + [
            mock.call("::: Train Scores (lemma) :::"),
            mock.call('+\tall acc:', 100),
            mock.call('+\tkno acc:', 101),
            mock.call('+\tunk acc:', 102)
        ]
        self.assertEqual(
            logger.__print_function__.call_args_list, expected,
            "It should print the first, each third (except the first) log and the last one"
        )

    def test_print_start(self):
        """ Ensure print works normally """
        logger = Logger(each=3, first=10, nb_epochs=100)
        logger.__print_function__ = mock.Mock()
        logger.write = mock.Mock()
        for i in range(100):
            i += 1
            logger.epoch(i, lambda: {"train_lemma": (0+i, 1+i, 2+i)})

        self.assertEqual(logger.__print_function__.called, True, "Calling to print should have been done")
        self.assertEqual(logger.write.called, False, "File has not been set and should not be called")
        self.assertEqual(len(logger.__print_function__.call_args_list), 41*4, "There should be 35 time the printing")
        expected = [
            call
            for i in range(1, 11)
            for call in [
                mock.call("::: Train Scores (lemma) :::"),
                mock.call('+\tall acc:', 0+i),
                mock.call('+\tkno acc:', 1+i),
                mock.call('+\tunk acc:', 2+i)
            ]
        ] + [
            call
            for i in range(4, 34)
            for call in [
                mock.call("::: Train Scores (lemma) :::"),
                mock.call('+\tall acc:', 0+i*3),
                mock.call('+\tkno acc:', 1+i*3),
                mock.call('+\tunk acc:', 2+i*3)
            ]
        ] + [
            mock.call("::: Train Scores (lemma) :::"),
            mock.call('+\tall acc:', 100),
            mock.call('+\tkno acc:', 101),
            mock.call('+\tunk acc:', 102)
        ]
        self.assertEqual(
            logger.__print_function__.call_args_list, expected,
            "It should print the first ten, each third (except the first ten) log and the last one"
        )

    def test_write(self):
        """ Ensure print works normally """
        logger = Logger(file="test_logs.csv", nb_epochs=100)
        logger.__print_function__ = mock.Mock()
        for i in range(100):
            i += 1
            logger.epoch(i, lambda: {"train_lemma": (0+i, 1+i, 2+i), "dev_lemma": (1+i, 2+i, 3+i)})

        self.assertEqual(logger.__print_function__.called, True, "Calling to print should have been done")

        self.assertEqual(
            logger.__print_function__.call_args_list,
            [
                call
                for i in range(1, 101)
                for call in [
                    mock.call("::: Train Scores (lemma) :::"),
                    mock.call('+\tall acc:', 0+i),
                    mock.call('+\tkno acc:', 1+i),
                    mock.call('+\tunk acc:', 2+i),
                    mock.call("::: Dev Scores (lemma) :::"),
                    mock.call('+\tall acc:', 1+i),
                    mock.call('+\tkno acc:', 2+i),
                    mock.call('+\tunk acc:', 3+i),
                ]
            ]
        )
        del logger
        logger_reader = Logger(file="test_logs.csv", nb_epochs=100)
        self.assertEqual(
            logger_reader.logs,
            [(i, {"train_lemma": (0+i, 1+i, 2+i), "dev_lemma": (1+i, 2+i, 3+i)}) for i in range(1, 101)],
            "Every line should be well written"
        )

    def test_write_on_loaded(self):
        """ Ensure print works normally """
        # Run a first time 100 epochs
        logger = Logger(file="test_logs.csv", nb_epochs=200)
        logger.__print_function__ = mock.Mock()
        for i in range(0, 100):
            i += 1
            logger.epoch(i, lambda: {"train_lemma": (0+i, 1+i, 2+i), "dev_lemma": (1+i, 2+i, 3+i)})
        del logger

        # Run a second time 100 epochs
        logger_second = Logger(shell=False, file="test_logs.csv", nb_epochs=200)
        logger_second.__print_function__ = mock.Mock()
        self.assertEqual(logger_second.__print_function__.called, False, "Print should not be called")
        self.assertEqual(
            logger_second.logs,
            [(i, {"train_lemma": (0+i, 1+i, 2+i), "dev_lemma": (1+i, 2+i, 3+i)}) for i in range(1, 101)],
            "Each first 100 lines should be well written"
        )
        for i in range(100, 200):
            i += 1
            logger_second.epoch(i, lambda: {"train_lemma": (0+i, 1+i, 2+i), "dev_lemma": (1+i, 2+i, 3+i)})
        del logger_second

        # Load and read
        logger_reader = Logger(shell=False, file="test_logs.csv", nb_epochs=200)
        self.assertEqual(
            logger_reader.logs,
            [(i, {"train_lemma": (0+i, 1+i, 2+i), "dev_lemma": (1+i, 2+i, 3+i)}) for i in range(1, 201)],
            "Every line + the old one should be well written"
        )
