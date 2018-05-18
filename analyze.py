# Script to compare tagged words to gold standard and analyze errors

N_WORDS = 20

import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--tagged_file", default="output/test_corpus.txt.csv")
parser.add_argument("--gold_file", default="/home/peter/box/POS-tagging/data/BrievenAlsBuit/test1000/test_corpus.tsv")
parser.add_argument("--gold_col_pos", type=int, default=1)
parser.add_argument("--gold_col_lemma", type=int, default=2)


args = parser.parse_args()

# Read file
df_tagged = pd.read_csv(args.tagged_file, sep="\t", header=0)
print(df_tagged)



