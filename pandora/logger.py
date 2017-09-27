import csv
import os
import ast


class Logger(object):
    """ Logger is a shared class that allows for centralizing efforts to print informations during training

    :param shell: Print training evaluation to shell
    :param file: Write training evaluation to a file
    :param each: Write or print each nth epoch
    :param nb_epochs: Total number of epochs
    :param first: Print/write the first nth epochs

    :cvar LABELS: Readable labels of the score_dict
    :cvar PRINT_ORDER: Order in which to print
    """
    LABELS = {
        "train_lemma": "Train Scores (lemma)",
        "dev_lemma": "Dev Scores (lemma)",
        "dev_lemma_postcorrect": "Dev scores (lemmas) -> postcorrected",
        "test_lemma": "Test scores (lemma)",
        "train_pos": "Train scores (pos)",
        "dev_pos": "Dev scores (pos)",
        "test_pos": "Test scores (pos)",
        "train_morph": "Train scores (morph)",
        "dev_morph": "Dev scores (morph)",
        "test_morph": "Test scores (morph)",
    }
    PRINT_ORDER = ["train_lemma", "dev_lemma", "dev_lemma_postcorrect", "test_lemma", "train_pos", "dev_pos",
                   "test_pos", "train_morph", "dev_morph", "test_morph"]
    SUBKEYS = ["all", "kno", "unk"]

    def __init__(self, shell=True, file=None, each=1, nb_epochs=1, first=1):

        self.shell = shell
        self.file = file
        self.each = each
        self.nb_epochs = nb_epochs
        self.first = first
        self.logs = []
        self.__print_function__ = print
        if self.file is not None:
            self.load()

    def load(self):
        """ Load former logs from CSV"""
        # If file exist, load it !
        if self.file is not None and os.path.isfile(self.file):
            with open(self.file) as f:
                csvfile = csv.DictReader(f)
                for row in csvfile:
                    simplified_row = {}
                    for key in set([k.split("#")[0] for k in row.keys() if k != "epoch"]):
                        simplified_row[key] = tuple(ast.literal_eval(row[key+"#"+subkey]) for subkey in Logger.SUBKEYS)
                    self.logs.append(
                        (int(row["epoch"]), simplified_row)
                    )

    def epoch(self, curr_epoch, callback):
        """ At the end of an epoch, decided wether or not to run the callback to get a score dict and then export it to
        due outputs

        :param curr_epoch: Current epoch (1 Based according to Tagger !)
        :param callback: Get Score dict function to call
        :return: Score dictionary
        """
        if curr_epoch in range(0, self.first+1) or curr_epoch == self.nb_epochs or curr_epoch % self.each == 0:
            score_dict = callback()
            self.logs.append((curr_epoch, score_dict))
            if self.shell is True:
                self.print()
            if isinstance(self.file, str):
                self.write()
            return score_dict
        return {}

    def print(self):
        """ Print to shell the current results
        """
        epoch_index, score_dict = self.logs[-1]
        for cat in [key for key in self.PRINT_ORDER if key in score_dict]:
            all_acc, kno_acc, unk_acc = score_dict[cat]
            self.__print_function__("::: {label} :::".format(label=self.LABELS[cat]))
            self.__print_function__('+\tall acc:', all_acc)
            self.__print_function__('+\tkno acc:', kno_acc)
            self.__print_function__('+\tunk acc:', unk_acc)

    def write(self):
        """ Write to CSV the current results
        """
        first_epoch, score_dict_epoch = self.logs[0]
        header = []
        for key in [k for k in self.PRINT_ORDER if k in score_dict_epoch]:
            header += [key+"#"+subkey for subkey in Logger.SUBKEYS]

        with open(self.file, "w") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch"]+header)
            writer.writeheader()
            for epoch_index, score_dict_epoch in self.logs:
                row = {"epoch": epoch_index}
                for key in score_dict_epoch:
                    for subkey, value in zip(Logger.SUBKEYS, score_dict_epoch[key]):
                        row[key+"#"+subkey] = value
                writer.writerow(row)

