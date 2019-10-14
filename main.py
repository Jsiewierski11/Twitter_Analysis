import numpy as np
import pandas as pd
from src.cleaner import Cleaner
from src.kmeans_operator import Kmeans_Operator as KMO

if __name__ == '__main__':
    twitter = pd.read_csv('data/full-corpus.csv', encoding='utf-8')
    corpus = twitter['TweetText'].to_numpy()


    '''
    Running LDA with Gensim
    '''
    # cleaner = Cleaner(corpus)
    
    # # # Using Gensim Stopwords
    # # cleaner.tokenize_corpus()
    # # word_count = cleaner.wc_whole_corpus()
    # # cleaner.plot_wc(word_count, filepath='media/tf_whole_corpus.png')

    # # Using custom Stopwords
    # cleaner.tokenize_corpus(custom_stopwords=True)
    # word_count = cleaner.wc_whole_corpus()
    # cleaner.plot_wc(word_count, filepath='media/tf_custom_sw.png')

    # cleaner.create_tdf()
    # # cleaner.print_tdf()
    # lda = cleaner.create_lda_model()
    # cleaner.print_top_words(lda)
    # cleaner.print_perplexity_coherence(lda)



    '''
    Running K-Means
    '''
    kmeans = KMO(n=4)
    documents = twitter['TweetText'].values
    documents = list(documents)
    
    kmeans.set_vec_doc_mat(documents)
    km_model = kmeans.create_and_fit()
    kmeans.print_terms_per_cluster(km_model)