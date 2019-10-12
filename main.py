import numpy as np
import pandas as pd
from src.cleaner import Cleaner

if __name__ == '__main__':
    twitter = pd.read_csv('data/full-corpus.csv', encoding='utf-8')
    corpus = twitter['TweetText'].to_numpy()

    cleaner = Cleaner(corpus)
    cleaner.tokenize_corpus()
    cleaner.create_tdf()
    cleaner.print_tdf()
    lda = cleaner.create_lda_model()
    cleaner.print_top_words(lda)