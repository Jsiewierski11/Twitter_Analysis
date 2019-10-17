import numpy as np
import pandas as pd
from src.cleaner import Cleaner
from src.kmeans_operator import Kmeans_Operator as KMO
from src.visualizer import Visualizer

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


def get_topics_corpus(df):
    apple_df = df[df['Topic'] == 'apple']
    google_df = df[df['Topic'] == 'google']
    ms_df = df[df['Topic'] == 'microsoft']
    twitter_df = df[df['Topic'] == 'twitter']
    apple_corpus = apple_df['TweetText'].to_numpy()
    google_corpus = google_df['TweetText'].to_numpy()
    ms_corpus = ms_df['TweetText'].to_numpy()
    twitter_corpus = twitter_df['TweetText'].to_numpy()
    return apple_corpus, google_corpus, ms_corpus, twitter_corpus


def run_lda(corpus, num_topics=4, custom_stopwords=False, filepath_wc=None, make_vis=True, filepath_lda=None):
    '''
    Running LDA with Gensim
    '''
    cleaner = Cleaner(corpus)
    viz = Visualizer()

    if custom_stopwords:
        # Using custom Stopwords
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
    apple_cleaner = Cleaner(apple_corpus)
    apple_cleaner.tokenize_corpus()
    apple_cleaner.create_bow()
    model_list, coherence_values = apple_cleaner.compute_coherence_values()
    viz.plot_coherence(model_list, coherence_values, filepath='media/apple_coherence3.png', color='grey', title='Apple')


    # Google
    google_cleaner = Cleaner(google_corpus)
    google_cleaner.tokenize_corpus()
    google_cleaner.create_bow()
    model_list, coherence_values = google_cleaner.compute_coherence_values()
    viz.plot_coherence(model_list, coherence_values, filepath='media/google_coherence3.png', color='orange', title='Google')


    # Microsoft
    ms_cleaner = Cleaner(ms_corpus)
    ms_cleaner.tokenize_corpus()
    ms_cleaner.create_bow()
    model_list, coherence_values = ms_cleaner.compute_coherence_values()
    viz.plot_coherence(model_list, coherence_values, filepath='media/microsoft_coherence3.png', color='blue', title='Microsoft')


    # Twiiter
    twitter_cleaner = Cleaner(twitter_corpus)
    twitter_cleaner.tokenize_corpus()
    twitter_cleaner.create_bow()
    model_list, coherence_values = twitter_cleaner.compute_coherence_values()
    viz.plot_coherence(model_list, coherence_values, filepath='media/twitter_coherence3.png', color='cyan', title='Twitter')


if __name__ == '__main__':
    twitter = pd.read_csv('data/full-corpus.csv', encoding='utf-8')
    viz = Visualizer()



    '''
    Segmenting df by Topics 'Apple, Google, Microsoft, Twitter'
    '''
    
    apple_corpus, google_corpus, ms_corpus, twitter_corpus = get_topics_corpus(twitter)


    '''
    Getting Coherence plots of all the Topics to determine number of clusters to use
    '''
    # plot_coherence_on_companies(twitter, viz)
    


    '''
    Running LDA on all the Topics
    '''
    print('Latent Topics for Tweets about Apple')
    run_lda(apple_corpus, num_topics=3, custom_stopwords=True, filepath_wc='media/tf_apple_mystop.png', make_vis=True, filepath_lda='media/apple_mystop.html')
    print('\n\n')

    print('Latent Topics for Tweets about Google')
    run_lda(google_corpus, num_topics=8, custom_stopwords=True, filepath_wc='media/tf_google_mystop.png', make_vis=True, filepath_lda='media/google_mystop.html')
    print('\n\n')

    print('Latent Topics for Tweets about Microsoft')
    run_lda(ms_corpus, num_topics=5, custom_stopwords=True, filepath_wc='media/tf_microsoft_mystop.png', make_vis=True, filepath_lda='media/microsoft_mystop.html')
    print('\n\n')

    print('Latent Topics for Tweets about Twitter')
    run_lda(twitter_corpus, num_topics=3, custom_stopwords=True, filepath_wc='media/tf_twitter_mystop.png', make_vis=True, filepath_lda='media/twitter_mystop.html')
    print('\n\n')
    
    
    
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


    
