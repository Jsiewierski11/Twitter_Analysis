import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os, sys
sys.path.append(os.path.abspath('..'))
from src.gensim_lda import Gensim_LDA
from src.kmeans_operator import Kmeans_Operator as KMO
from src.visualizer import Visualizer
from src.naive_bayes import Naive_Bayes
from src.df_cleaner import DF_Cleaner
from src.runner import Runner

# FIXME: This code needs major refactoring. This is holdover code from topic modeling 
#        project that needs to be udated to current code base and needs to be made 
#        more modular.


def run_lda(corpus, num_topics=4, custom_stopwords=False, filepath_wc=None, make_vis=True, filepath_lda=None):
    '''
    Running LDA with Gensim
    '''
    cleaner = Gensim_LDA(corpus)
    viz = Visualizer()

    if custom_stopwords:
        # Using custom StCleaner(opwords
        cleaner.tokenize_corpus(custom_stopwords=True)
        word_count = cleaner.wc_whole_corpus()
        if filepath_wc is None:
            viz.plot_wc(word_count, filepath='media/tf_custom_sw.png')
        else:
            viz.plot_wc(word_count, filepath=filepath_wc)
    else:
        # Using Gensim Stopwords
        cleaner.tokenize_corpus()
        word_count = cleaner.wc_whole_corpus()
        if filepath_wc is None:
            viz.plot_wc(word_count, filepath='media/tf_whole_corpus.png')
        else:
            viz.plot_wc(word_count, filepath=filepath_wc)

    cleaner.create_bow()
    lda = cleaner.create_lda_model(num_topics=num_topics)
    cleaner.print_top_words(lda)
    cleaner.print_perplexity_coherence(lda)
    if make_vis:
        viz.make_pyLDAvis(lda, cleaner.bow, cleaner.id2word, filepath=filepath_lda)

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


def plot_coherence_on_companies(twitter, viz):
    # Apple
    apple_cleaner = Gensim_LDA(apple_corpus)
    apple_cleaner.tokenize_corpus()
    apple_cleaner.create_bow()
    model_list, coherence_values = apple_cleaner.compute_coherence_values()
    viz.plot_coherence(model_list, coherence_values, filepath='media/apple_coherence3.png', color='grey', title='Apple')


    # Google
    google_cleaner = Gensim_LDA(google_corpus)
    google_cleaner.tokenize_corpus()
    google_cleaner.create_bow()
    model_list, coherence_values = google_cleaner.compute_coherence_values()
    viz.plot_coherence(model_list, coherence_values, filepath='media/google_coherence3.png', color='orange', title='Google')


    # Microsoft
    ms_cleaner = Gensim_LDA(ms_corpus)
    ms_cleaner.tokenize_corpus()
    ms_cleaner.create_bow()
    model_list, coherence_values = ms_cleaner.compute_coherence_values()
    viz.plot_coherence(model_list, coherence_values, filepath='media/microsoft_coherence3.png', color='blue', title='Microsoft')


    # Twiiter
    twitter_cleaner = Gensim_LDA(twitter_corpus)
    twitter_cleaner.tokenize_corpus()
    twitter_cleaner.create_bow()
    model_list, coherence_values = twitter_cleaner.compute_coherence_values()
    viz.plot_coherence(model_list, coherence_values, filepath='media/twitter_coherence3.png', color='cyan', title='Twitter')


if __name__ == '__main__':

    # twitter = pd.read_csv('data/full-corpus.csv', encoding='utf-8')
    # # corpus = twitter['TweetText'].to_numpy()
    # viz = Visualizer()
    # # gen_lda = Gensim_LDA(corpus)

    '''
    Getting Coherence of Whole Dataset
    '''
    # cleaner = Cleaner(corpus)
    # cleaner.tokenize_corpus(custom_stopwords=True)
    # cleaner.create_bow()
    # model_list, coherence_values = cleaner.compute_coherence_values()
    # viz.plot_coherence(model_list, coherence_values, color='Black', title='Whole Corpus')



    '''
    Segmenting df by Topics 'Apple, Google, Microsoft, Twitter'
    '''
    
    # apple_corpus, google_corpus, ms_corpus, twitter_corpus = get_topics_corpus(twitter)
    # apple_df, google_df, ms_df, twitter_df = get_topics_df(twitter)


    '''
    Creating Pie Charts of Sentiments
    '''

    # viz.plot_sentiments_pie(apple_df, title='Apple', filepath='media/apple_sentiments.png')
    # viz.plot_sentiments_pie(google_df, title='Google', filepath='media/google_sentiments.png')
    # viz.plot_sentiments_pie(ms_df, title='Microsoft', filepath='media/microsoft_sentiments.png')
    # viz.plot_sentiments_pie(twitter_df, title='Twitter', filepath='media/twitter_sentiments.png')

    '''
    Getting Coherence plots of all the Topics to determine number of clusters to use
    '''
    # plot_coherence_on_companies(twitter, viz)



    '''
    Running K-Means
    '''
    # print('Clusters for K-Means on whole corpus')
    # run_kmeans(corpus)
    # print('\n\n')

    # print('Clusters for K-Means on apple corpus')
    # run_kmeans(apple_corpus)
    # print('\n\n')

    # print('Clusters for K-Means on google corpus')
    # run_kmeans(google_corpus)
    # print('\n\n')


    # print('Clusters for K-Means on microsoft corpus')
    # run_kmeans(ms_corpus)
    # print('\n\n')

    # print('Clusters for K-Means on twitter corpus')
    # run_kmeans(twitter_corpus)
    # print('\n\n')
    


    '''
    Running LDA on all the Topics
    '''
    # print('Latent Topics for All Documents LDA')
    # run_lda(corpus, num_topics=4, custom_stopwords=True, filepath_wc='media/tf_4_whole_corpus.png', make_vis=True, filepath_lda='media/4_whole_corpus.html')
    # print('\n\n')

    # print('Latent Topics for Tweets about Apple')
    # run_lda(apple_corpus, num_topics=5, custom_stopwords=True, filepath_wc='media/tf_apple_mystop.png', make_vis=True, filepath_lda='media/apple_mystop.html')
    # print('\n\n')

    # print('Latent Topics for Tweets about Google')
    # run_lda(google_corpus, num_topics=3, custom_stopwords=True, filepath_wc='media/tf_google_mystop.png', make_vis=True, filepath_lda='media/google_mystop.html')
    # print('\n\n')

    # print('Latent Topics for Tweets about Microsoft')
    # run_lda(ms_corpus, num_topics=5, custom_stopwords=True, filepath_wc='media/tf_microsoft_mystop.png', make_vis=True, filepath_lda='media/microsoft_mystop.html')
    # print('\n\n')

    # print('Latent Topics for Tweets about Twitter')
    # run_lda(twitter_corpus, num_topics=3, custom_stopwords=True, filepath_wc='media/tf_twitter_mystop.png', make_vis=True, filepath_lda='media/twitter_mystop.html')
    # print('\n\n')
    
    
    
    '''
    Segmenting Based off of Sentiment
    '''
    # corpus = twitter['TweetText'].to_numpy()
    # pos_corpus, neg_corpus, neutral_corpus, irr_corpus = get_sentiment_corpus(twitter)

    # run_all_models(corpus, pos_corpus, neg_corpus, neutral_corpus, irr_corpus)
    # cleaner, lda_model = run_lda(corpus, num_topics=13, custom_stopwords=True, filepath='media/tf_custom_sw.png', make_vis=True)
    # cleaner.plot_coherence()

    # cleaner = Cleaner(corpus)
    # cleaner.tokenize_corpus(custom_stopwords=True)
    # cleaner.create_bow()
    # lda = cleaner.create_lda_model(num_topics=12)
    
    
    # Testing Visualizer functions
    # viz.plot_sentiments_pie()
    # viz.plot_categories_bar()
    # viz.plot_categories_pie()

    # word_count = cleaner.wc_whole_corpus()
    # viz.plot_wc(word_count, n=20, filepath='media/tf_withviz.png')

    # model_list, coherence_values, u_mass_vals = cleaner.compute_coherence_values()
    # viz.plot_coherence(model_list, coherence_values, u_mass_vals)

    # viz.make_pyLDAvis(lda, cleaner.bow, cleaner.id2word, filepath='media/LDA_12_topics.html')

    # cleaner.document_topic_distribution(lda_model)
    # cleaner.determine_doc_topic(corpus, 50)