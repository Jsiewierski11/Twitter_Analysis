import numpy as np
import pandas as pd
from src.cleaner import Cleaner

if __name__ == '__main__':
    twitter = pd.read_csv('data/full-corpus.csv', encoding='utf-8')
    corpus = twitter['TweetText'].to_numpy()

    cleaner = Cleaner(corpus)
    
    # # Using Gensim Stopwords
    # cleaner.tokenize_corpus()
    # word_count = cleaner.wc_whole_corpus()
    # cleaner.plot_wc(word_count, filepath='media/tf_whole_corpus.png')

    # Using custom Stopwords
    cleaner.tokenize_corpus(custom_stopwords=True)
    word_count = cleaner.wc_whole_corpus()
    cleaner.plot_wc(word_count, filepath='media/tf_custom_sw.png')

    cleaner.create_tdf()
    # cleaner.print_tdf()
    lda = cleaner.create_lda_model()
    cleaner.print_top_words(lda)
    cleaner.print_perplexity_coherence(lda)