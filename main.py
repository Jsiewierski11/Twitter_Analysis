import numpy as np
import pandas as pd
from src.cleaner import Cleaner
from src.kmeans_operator import Kmeans_Operator as KMO

def get_sentiment_corpus(df):
    pos_df = df[df['Sentiment'] == 'positive']
    neg_df = df[df['Sentiment'] == 'negative']
    neutral_df = df[df['Sentiment'] == 'neutral']
    irr_df = df[df['Sentiment'] == 'irrelevant']
    pos_corpus = pos_df['TweetText'].to_numpy()
    neg_corpus = neg_df['TweetText'].to_numpy()
    neutral_corpus = neutral_df['TweetText'].to_numpy()
    irr_corpus = irr_df['TweetText'].to_numpy()
    return pos_corpus, neg_corpus, neutral_corpus, irr_corpus


def run_lda(corpus, custom_stopwords=False, filepath=None, make_vis=False):
    '''
    Running LDA with Gensim
    '''
    cleaner = Cleaner(corpus)

    if custom_stopwords:
        # Using custom Stopwords
        cleaner.tokenize_corpus(custom_stopwords=True)
        word_count = cleaner.wc_whole_corpus()
        if filepath is None:
            cleaner.plot_wc(word_count, filepath='media/tf_custom_sw.png')
        else:
            cleaner.plot_wc(word_count, filepath=filepath)
    else:
        # Using Gensim Stopwords
        cleaner.tokenize_corpus()
        word_count = cleaner.wc_whole_corpus()
        if filepath is None:
            cleaner.plot_wc(word_count, filepath='media/tf_whole_corpus.png')
        else:
            cleaner.plot_wc(word_count, filepath=filepath)

    cleaner.create_tdf()
    # cleaner.print_tdf()
    lda = cleaner.create_lda_model()
    cleaner.print_top_words(lda)
    cleaner.print_perplexity_coherence(lda) 
    if make_vis:
        cleaner.make_pyLDAvis(lda) 

    return cleaner, lda



def run_kmeans(corpus, remove_stopwords=False):
    '''
    Running K-Means
    '''
    kmeans = KMO(n=4)
    if remove_stopwords:
        kmeans.set_vec_doc_mat(corpus, remove_stopwords=remove_stopwords)   
    else:
        kmeans.set_vec_doc_mat(corpus)
    km_model = kmeans.create_and_fit()
    kmeans.print_terms_per_cluster(km_model)


def run_all_models(corpus, pos_corpus, neg_corpus, neutral_corpus, irr_corpus):
    '''
    Running Models
    '''
    # K-Means
    print('Latent Topics for All Documents K-Means')
    run_kmeans(corpus, remove_stopwords=True)
    print('\n\n')
    print('Latent Topics for Positive Documents K-Means')
    run_kmeans(pos_corpus, remove_stopwords=True)
    print('\n\n')
    print('Latent Topics for Negative Documents K-Means')
    run_kmeans(neg_corpus, remove_stopwords=True)
    print('\n\n')


    # LDA
    print('Latent Topics for All Documents LDA')
    run_lda(corpus, custom_stopwords=True, filepath='media/tf_custom_sw.png')
    print('\n\n')
    
    print('Latent Topics for Positive Documents LDA')
    run_lda(pos_corpus, custom_stopwords=True, filepath='media/pos_tf.png')
    print('\n\n')

    print('Latent Topics for Negative Documents LDA')
    run_lda(neg_corpus, custom_stopwords=True, filepath='media/neg_tf.png')
    print('\n\n')  

    print('Latent Topics for Neutral Documents LDA')
    run_lda(neutral_corpus, custom_stopwords=True, filepath='media/neutral_tf.png')
    print('\n\n')    

    print('Latent Topics for Irrelevant Documents LDA')
    run_lda(irr_corpus, custom_stopwords=True, filepath='media/irrelevant_tf.png')
    print('\n\n')  


if __name__ == '__main__':
    twitter = pd.read_csv('data/full-corpus.csv', encoding='utf-8')
    
    corpus = twitter['TweetText'].to_numpy()
    pos_corpus, neg_corpus, neutral_corpus, irr_corpus = get_sentiment_corpus(twitter)

    # run_all_models(corpus, pos_corpus, neg_corpus, neutral_corpus, irr_corpus)
    cleaner, lda_model = run_lda(corpus, custom_stopwords=True, filepath='media/tf_custom_sw.png')

    cleaner.document_topic_distribution(lda_model)
    cleaner.determine_doc_topic(corpus, 50)