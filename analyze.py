# Script to compare tagged words to gold standard and analyze errors

N_WORDS = 20
FORMAT = "stanford"

import argparse
import pandas as pd
import numpy as np


def error_analysis(df_comb, var, var_gold):
    # Analyze wrong lemma results
    df_wrong = df_comb[df_comb[var]!=df_comb[var_gold]]
    df_wrong = df_wrong[["tokens", var, var_gold]]
    # Sort by most frequent combination
    return df_wrong.groupby(["tokens", var, var_gold]).size().sort_values(ascending=False)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tagged_file", default="output/test_corpus.txt.tsv")
    parser.add_argument("--gold_file", default="/home/peter/box/POS-tagging/data/BrievenAlsBuit/test10000/test_corpus.tsv")
    parser.add_argument("--gold_col_tokens", type=int, default=0)
    parser.add_argument("--gold_col_pos", type=int, default=1)
    parser.add_argument("--gold_col_lemma", type=int, default=2)
    parser.add_argument("--format", default=FORMAT, choices=["pandora", "stanford"])
    
    
    args = parser.parse_args()
    
    # Read tagged file
    if args.format=="pandora":
        df_tagged = pd.read_csv(args.tagged_file, sep="\s{2,}|\t+", engine="python", header=0)
        df_tagged = df_tagged.drop(columns=["lemmas"])
        df_tagged = df_tagged.rename(columns={"postcorrect_lemmas":"lemmas"})
    else:
        df_tagged = pd.read_csv(args.tagged_file, sep="\s{2,}|\t+", engine="python", header=None)
        df_tagged = df_tagged.rename(columns={0:"tokens", 1: "pos"}) # TODO: parametrize
    # Read gold file
    # Match tab or multiple whitespaces as delimiter
    df_gold = pd.read_csv(args.gold_file, sep="\s{2,}|\t+", engine="python", header=None)
    df_gold = df_gold.rename(columns={df_gold.columns[args.gold_col_tokens]: "tokens", df_gold.columns[args.gold_col_pos]: "pos_gold", df_gold.columns[args.gold_col_lemma]: "lemmas_gold" })
    
    # Check if gold and tagged file contain same words
    if not df_tagged["tokens"].equals(df_gold["tokens"]):
        print(set(df_tagged["tokens"]) - set(df_gold["tokens"]))
        print(set(df_gold["tokens"]) - set(df_tagged["tokens"]))
        raise IOError("Gold and tagged file do not contain same tokens!")
    
    # Combine dataframes
    df_comb = pd.concat([df_tagged,df_gold],axis=1)
    df_comb = df_comb.loc[:,~df_comb.columns.duplicated()]
    
    # Compute lemmatization and POS tagging accuracy
    total = len(df_comb)
    correct_pos_count = len(df_comb[df_comb["pos"]==df_comb["pos_gold"]])
    acc_pos = correct_pos_count / total
    print("POS tagging accuracy: " + str(acc_pos))
    # Show analysis of most frequent errors
    print(error_analysis(df_comb,"pos", "pos_gold"))
    
    # If format is Pandora, also include lemmas
    if args.format=="pandora":    
        correct_lemma_count = len(df_comb[df_comb["lemmas"]==df_comb["lemmas_gold"]])
        correct_both_count = len(df_comb[(df_comb["pos"]==df_comb["pos_gold"]) & (df_comb["lemmas"]==df_comb["lemmas_gold"])])
        
        acc_lemma = correct_lemma_count / total
        
        acc_both = correct_both_count / total
        print("Lemmatization accuracy: " + str(acc_lemma))
        
        print("POS&lemma accuracy: " + str(acc_both) + ". Expected (pos*lemma): " + str(acc_lemma*acc_pos))
        
        # Show analysis of most frequent errors
        print(error_analysis(df_comb,"lemmas", "lemmas_gold"))
    

if __name__ == "__main__":
    main()

