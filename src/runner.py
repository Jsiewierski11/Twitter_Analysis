import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.gensim_lda import Gensim_LDA
from src.kmeans_operator import Kmeans_Operator as KMO
from src.visualizer import Visualizer
from src.naive_bayes import Naive_Bayes
from src.my_doc2vec import My_Doc2Vec
from src.df_cleaner import DF_Cleaner
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB


class Runner(object):
    '''
    This class is used to run various aspects of the project.
    Each function was written so it could be run independantely and still produce the desired results.
    '''
    
    def __init__(self):
        pass

    def run_naive_bayes_sentiment(self):
        '''
        This function takes no inputs and returns nothing.
        Function will:
            - Load data to pandas dataframe.
            - Balance the corpus so that there is an equal amount of tweets for each sentiment and drop tweets labeled with irrelevant sentiment.
            - Perform train test split on the dataset.
            - Perform preprocessing on the text and create TF-IDF array of the corpus.
            - Train Naive Bayes model for sentiment classification on TF-IDF and save as a .pkl file.
            - Print performance metrics to the console and save a .png file of Confusion Matrix.
        '''

        print("Running Naive Bayes Classification with TF-IDF")
        twitter = pd.read_csv('../data/full-corpus.csv', encoding='utf-8')
        viz = Visualizer()
        nb = Naive_Bayes()
        dfc = DF_Cleaner()
        
        '''
        Sentiment Classification with Naive Bayes
        '''
        pos_df, neg_df, neutral_df, irr_df = dfc.get_sentiment_df(twitter)
        balanced_df = dfc.balance_df([neg_df, neutral_df], pos_df)
        y = balanced_df.pop('Sentiment')
        X_train, X_test, y_train, y_test = train_test_split(balanced_df, y, random_state=42)

        train_text = X_train['TweetText'].to_numpy()
        test_text = X_test['TweetText'].to_numpy()

        X_train_counts, X_train_tfidf = nb.compute_tf_and_tfidf(train_text, ngram_range=(1, 5))
        y_pred = nb.classify(X_train_tfidf, y_train, test_text)

        nb.print_metrics(y_test, y_pred)
        nb.pickle_model(filepath_cv='../models/count_vect_sent.pkl', filepath_clf='../models/naive_bayes_sent.pkl')
        viz.plot_confusion_matrix(y_test, y_pred, classes=['positive', 'negative', 'neutral'], \
                                  title='Multinomial Naive Bayes with TF-IDF')
        plt.savefig('../media/confusion_matrix/tfidf_nb_confmat_sentiment.png')
        plt.close()
        print('\n\n')

    
    def run_naive_bayes_topic(self):
        '''
        This function takes no inputs and returns nothing.
        Function will:
            - Load the corpus to a pandas dataframe.
            - Perform train test split on the dataset.
            - Perform preprocessing on the text and create TF-IDF array of the corpus.
            - Train Naive Bayes model for topic classification on TF-IDF and save as a .pkl file to the models directory.
            - Print performance metrics to the console and save a .png file of Confusion Matrix.
        '''

        print("Running Naive Bayes Classification with TF-IDF")
        twitter = pd.read_csv('../data/full-corpus.csv', encoding='utf-8')
        viz = Visualizer()
        nb = Naive_Bayes()
        dfc = DF_Cleaner()
        
        '''
        Topic Classification with Naive Bayes
        '''
        y = twitter.pop('Topic')
        X_train, X_test, y_train, y_test = train_test_split(twitter, y, random_state=42)

        train_text = X_train['TweetText'].to_numpy()
        test_text = X_test['TweetText'].to_numpy()

        X_train_counts, X_train_tfidf = nb.compute_tf_and_tfidf(train_text)
        y_pred = nb.classify(X_train_tfidf, y_train, test_text)
        nb.print_metrics(y_test, y_pred)
        nb.pickle_model(filepath_cv='../models/count_vect_companies.pkl', filepath_clf='../models/naive_bayes_companies.pkl')
        viz.plot_confusion_matrix(y_test, y_pred, classes=['apple', 'google', 'microsoft', 'twitter'], \
                                  title='Multinomial Naive Bayes with TF-IDF')
        plt.savefig('../media/confusion_matrix/tfidf_nb_confmat_companies.png')
        plt.close()
        print('\n\n')


    def run_doc2vec_logreg(self):
        '''
        This function takes no inputs and returns nothing.
        Function will:
            - Load data to pandas dataframe.
            - Balance the corpus so that there is an equal amount of tweets for each sentiment and drop tweets labeled with irrelevant sentiment.
            - Perform train test split on the dataset.
            - Perform preprocessing on the text and create Doc2Vec array of the corpus.
            - Train Logistic Regression model for sentiment classification on Doc2Vec and save as a .pkl file to the models directory.
            - Print performance metrics to the console and save a .png file of Confusion Matrix.
        '''

        print("Running Logistic Regression Classification with Doc2Vec")
        twitter = pd.read_csv('../data/full-corpus.csv', encoding='utf-8')
        dfc = DF_Cleaner()
        viz = Visualizer()
        d2v = My_Doc2Vec()

        # Balancing and Train Test Split
        pos_df, neg_df, neutral_df, irr_df = dfc.get_sentiment_df(twitter)
        balanced_df = dfc.balance_df([neg_df, neutral_df], pos_df)
        train, test = train_test_split(balanced_df, test_size=0.3, random_state=42) 

        '''
        Sentiment Classification with Logistic Regression and Doc2Vec
        '''
        test_tagged, train_tagged = d2v.tag_doc(test, train)
        d2v.create_model_and_vocab(train_tagged)
        d2v.train_model(test_tagged, train_tagged)

        y_train, X_train = d2v.vec_for_learning(train_tagged)
        y_test, X_test = d2v.vec_for_learning(test_tagged)
        logreg = LogisticRegression(n_jobs=1, C=1e5)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        
        print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
        print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

        d2v.pickle_model(logreg, filepath='../models/doc2vec_logreg.pkl')
        viz.plot_confusion_matrix(y_test, y_pred, classes=['positive', 'negative', 'neutral', 'irrelevant'], \
                                  title='Logestic Regression with Doc2Vec')
        plt.savefig('../media/confusion_matrix/d2v_logreg_confmat.png')
        plt.close()

    def run_doc2vec_naivebayes(self):
        '''
        This function takes no inputs and returns nothing.
        Function will:
            - Load data to pandas dataframe.
            - Balance the corpus so that there is an equal amount of tweets for each sentiment and drop tweets labeled with irrelevant sentiment.
            - Perform train test split on the dataset.
            - Perform preprocessing on the text and create Doc2Vec array of the corpus.
            - Train Naive Bayes model for sentiment classification on Doc2Vec and save as a .pkl file to the models directory.
            - Print performance metrics to the console and save a .png file of Confusion Matrix.
        '''

        print("Running Naive Bayes Classification with Doc2Vec")
        twitter = pd.read_csv('../data/full-corpus.csv', encoding='utf-8')
        dfc = DF_Cleaner()
        viz = Visualizer()

        # Balancing and Train Test Split
        pos_df, neg_df, neutral_df, irr_df = dfc.get_sentiment_df(twitter)
        balanced_df = dfc.balance_df([neg_df, neutral_df], pos_df)
        train, test = train_test_split(balanced_df, test_size=0.3, random_state=42) 

        '''
        Sentiment Classification with Naive Bayes and Doc2Vec
        '''
        d2v = My_Doc2Vec()
        test_tagged, train_tagged = d2v.tag_doc(test, train)
        d2v.create_model_and_vocab(train_tagged)
        d2v.train_model(test_tagged, train_tagged)

        y_train, X_train = d2v.vec_for_learning(train_tagged)
        y_test, X_test = d2v.vec_for_learning(test_tagged)

        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
        print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

        d2v.pickle_model(clf, filepath='../models/doc2vec_naive_bayes.pkl')
        viz.plot_confusion_matrix(y_test, y_pred, classes=['positive', 'negative', 'neutral', 'irrelevant'], \
                                  title='Guassian Navie Bayes with Doc2Vec')
        plt.savefig('../media/confusion_matrix/d2v_nb_confmat.png')
        plt.close()


    def make_plots(self):
        '''
        This function takes no inputs and returns nothing.
        Function will:
            - Load data to pandas dataframe.
            - Create bar chart of 20 most common words in the corpus.
            - Create word clouds of words relating to tweets for all the different sentiments, all topics, and the whole corpus.
            - Create bar chart of the number of tweets labeled with each sentiment.
            - Create bar chart of the number of tweets labeled with each topic.
            - All plots produced are saved as .png files to the media directory in their appropriate subdirectories.
        '''

        print("Creating Plots of the data")
        twitter = pd.read_csv('../data/full-corpus.csv', encoding='utf-8')
        viz = Visualizer()
        dfc = DF_Cleaner()

        pos_df, neg_df, neutral_df, irr_df = dfc.get_sentiment_df(twitter)
        apple_df, google_df, ms_df, twitter_df = dfc.get_topics_df(twitter)


        # Remove stop words and perform lemmatization to create Pandas Series
        processed_docs = twitter['TweetText'].apply(lambda x: dfc.preprocess(x, remove_common=False))
        processed_pos = pos_df['TweetText'].apply(lambda x: dfc.preprocess(x, remove_common=True))
        processed_neg = neg_df['TweetText'].apply(lambda x: dfc.preprocess(x, remove_common=True))
        processed_neutral = neutral_df['TweetText'].apply(lambda x: dfc.preprocess(x, remove_common=True))
        processed_apple = apple_df['TweetText'].apply(lambda x: dfc.preprocess(x, remove_common=False))
        processed_google = google_df['TweetText'].apply(lambda x: dfc.preprocess(x, remove_common=False))
        processed_ms = ms_df['TweetText'].apply(lambda x: dfc.preprocess(x, remove_common=False))
        processed_twitter = twitter_df['TweetText'].apply(lambda x: dfc.preprocess(x, remove_common=False))


        # Converting Pandas Series to numpy array
        doc_array = processed_docs.to_numpy()
        pos_doc = processed_pos.to_numpy()
        neg_doc = processed_neg.to_numpy()
        neutral_doc = processed_neutral.to_numpy()
        apple_doc = processed_apple.to_numpy()
        google_doc = processed_google.to_numpy()
        ms_doc = processed_ms.to_numpy()
        twitter_doc = processed_twitter.to_numpy()

        
        # Creating dictionary of word counts
        word_counts = dfc.wc_corpus(doc_array)
        pos_wordcounts = dfc.wc_corpus(pos_doc)
        neg_wordcounts = dfc.wc_corpus(neg_doc)
        neutral_wordcounts = dfc.wc_corpus(neutral_doc)

        
        # Converting Corpus numpy array to one giant string for word cloud
        big_string = dfc.doc_array_to_str(doc_array)
        pos_string = dfc.doc_array_to_str(pos_doc)
        neg_string = dfc.doc_array_to_str(neg_doc)
        neutral_string = dfc.doc_array_to_str(neutral_doc)
        apple_string = dfc.doc_array_to_str(apple_doc)
        google_string = dfc.doc_array_to_str(google_doc)
        ms_string = dfc.doc_array_to_str(ms_doc)
        twitter_string = dfc.doc_array_to_str(twitter_doc)


        print("creating bar plot of word counts")
        viz.plot_wc(word_counts, filepath='../media/tf/tf_whole_corpus.png', title='20 Most Common Words in Corpus')

        print("creating word clouds")
        viz.plot_wordcloud(big_string, title="All Tweets", filepath="../media/tf/word_cloud_all_tweets.png")
        viz.plot_wordcloud(pos_string, title="Positive Tweets", filepath="../media/tf/word_cloud_pos_tweets.png")
        viz.plot_wordcloud(neg_string, title="Negative Tweets", filepath="../media/tf/word_cloud_neg_tweets.png")
        viz.plot_wordcloud(neutral_string, title="Neutral Tweets", filepath="../media/tf/word_cloud_neutral_tweets.png")
        viz.plot_wordcloud(apple_string, title="Apple Tweets", filepath="../media/tf/word_cloud_apple_tweets.png")
        viz.plot_wordcloud(google_string, title="Google Tweets", filepath="../media/tf/word_cloud_google_tweets.png")
        viz.plot_wordcloud(ms_string, title="Microsoft Tweets", filepath="../media/tf/word_cloud_ms_tweets.png")
        viz.plot_wordcloud(twitter_string, title="Twitter Tweets", filepath="../media/tf/word_cloud_twitter_tweets.png")

        print("creating bar plot of sentiments")
        viz.plot_sentiments_bar()

        print("creating bar plot of categories")
        viz.plot_categories_bar()
        print('\n\n')