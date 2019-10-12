import numpy as np
import pandas as pd
from src.cleaner import Cleaner

if __name__ == '__main__':
    twitter = pd.read_csv('data/full-corpus.csv', encoding='utf-8')
    corpus = twitter['TweetText'].to_numpy()

    cleaner = Cleaner(corpus)
    print(type(cleaner.corpus))
    cleaner.tokenize_corpus()
    print(type(cleaner.corpus))

    # print(cleaner.corpus)