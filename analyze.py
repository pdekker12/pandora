# Script to compare tagged words to gold standard and analyze errors

N_WORDS = 20

import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--tagged_file", default="output/test_corpus.txt.tsv")
parser.add_argument("--gold_file", default="/home/peter/box/POS-tagging/data/BrievenAlsBuit/test10000/test_corpus.tsv")
parser.add_argument("--gold_col_tokens", type=int, default=0)
parser.add_argument("--gold_col_pos", type=int, default=1)
parser.add_argument("--gold_col_lemma", type=int, default=2)


args = parser.parse_args()

# Read tagged file
df_tagged = pd.read_csv(args.tagged_file, sep="\t", header=0)
# Read gold file
# Match tab or multiple whitespaces as delimiter
df_gold = pd.read_csv(args.gold_file, sep="\s{2,}|\t+", engine="python", header=None)
df_gold = df_gold.rename(columns={df_gold.columns[args.gold_col_tokens]: "tokens", df_gold.columns[args.gold_col_pos]: "pos", df_gold.columns[args.gold_col_lemma]: "lemmas" })


# Check if gold and tagged file contain same words
if not df_tagged["tokens"].equals(df_gold["tokens"]):
    raise IOError("Gold and tagged file do not contain same tokens!")

# Compute lemmatization and POS tagging accuracy
correct_lemma_count = len(df_tagged[df_tagged["lemmas"]==df_gold["lemmas"]])
correct_pos_count = len(df_tagged["lemmas"][df_tagged["pos"]==df_gold["pos"]])
total = len(df_tagged)
acc_lemma = correct_lemma_count / total
acc_pos = correct_pos_count / total
print("Lemmatization accuracy: " + str(acc_lemma))
print("POS tagging accuracy: " + str(acc_pos))



