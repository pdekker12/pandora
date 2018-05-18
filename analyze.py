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
df_tagged = df_tagged.drop(columns=["lemmas"])
df_tagged = df_tagged.rename(columns={"postcorrect_lemmas":"lemmas"})
# Read gold file
# Match tab or multiple whitespaces as delimiter
df_gold = pd.read_csv(args.gold_file, sep="\s{2,}|\t+", engine="python", header=None)
df_gold = df_gold.rename(columns={df_gold.columns[args.gold_col_tokens]: "tokens", df_gold.columns[args.gold_col_pos]: "pos_gold", df_gold.columns[args.gold_col_lemma]: "lemmas_gold" })

# Check if gold and tagged file contain same words
if not df_tagged["tokens"].equals(df_gold["tokens"]):
    raise IOError("Gold and tagged file do not contain same tokens!")

# Combine dataframes
df_comb = pd.concat([df_tagged,df_gold],axis=1)
df_comb = df_comb.loc[:,~df_comb.columns.duplicated()]

# Compute lemmatization and POS tagging accuracy
correct_lemma_count = len(df_comb[df_comb["lemmas"]==df_comb["lemmas_gold"]])
correct_pos_count = len(df_comb[df_comb["pos"]==df_comb["pos_gold"]])
correct_both_count = len(df_comb[(df_comb["pos"]==df_comb["pos_gold"]) & (df_comb["lemmas"]==df_comb["lemmas_gold"])])
total = len(df_comb)
acc_lemma = correct_lemma_count / total
acc_pos = correct_pos_count / total
acc_both = correct_both_count / total
print("Lemmatization accuracy: " + str(acc_lemma))
print("POS tagging accuracy: " + str(acc_pos))
print("POS&lemma accuracy: " + str(acc_both) + ". Expected (pos*lemma): " + str(acc_lemma*acc_pos))


# Analyze wrong lemma results
df_wrong_lemma = df_comb[df_comb["lemmas"]!=df_comb["lemmas_gold"]]
df_wrong_lemma = df_wrong_lemma[["tokens", "lemmas", "lemmas_gold"]]
# Sort by most frequent combination
print(df_wrong_lemma.groupby(['lemmas', 'lemmas_gold']).size().sort_values(ascending=False))

# Analyze wrong POS results
df_wrong_pos = df_comb[df_comb["pos"]!=df_comb["pos_gold"]]
df_wrong_pos = df_wrong_pos[["tokens", "pos", "pos_gold"]]
print(df_wrong_pos)