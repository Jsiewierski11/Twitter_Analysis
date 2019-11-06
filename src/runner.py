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
    def __init__(self):
        pass

    def run_naive_bayes_sentiment(self):
        print("Running Naive Bayes Classification with TF-IDF")
        twitter = pd.read_csv('../data/full-corpus.csv', encoding='utf-8')
        viz = Visualizer()
        
        '''
        Sentiment Classification with Naive Bayes
        '''
        nb = Naive_Bayes()
        dfc = DF_Cleaner()
        pos_df, neg_df, neutral_df, irr_df = dfc.get_sentiment_df(twitter)
        balanced_df = dfc.balance_df([neg_df, neutral_df], pos_df)
        y = balanced_df.pop('Sentiment')
        X_train, X_test, y_train, y_test = train_test_split(balanced_df, y, random_state=42)

        train_text = X_train['TweetText'].to_numpy()
        test_text = X_test['TweetText'].to_numpy()

        X_train_counts, X_train_tfidf = nb.compute_tf_and_tfidf(train_text)
        y_pred = nb.classify(X_train_tfidf, y_train, test_text)
        nb.print_metrics(y_test, y_pred)
        nb.pickle_model(filepath_cv='../models/count_vect_sent.pkl', filepath_clf='../models/naive_bayes_sent.pkl')
        viz.plot_confusion_matrix(y_test, y_pred, classes=['positive', 'negative', 'neutral'], \
                                  title='Multinomial Naive Bayes with TF-IDF')
        plt.savefig('../media/confusion_matrix/tfidf_nb_confmat_sentiment.png')
        plt.close()

    
    def run_naive_bayes_topic(self):
        print("Running Naive Bayes Classification with TF-IDF")
        twitter = pd.read_csv('../data/full-corpus.csv', encoding='utf-8')
        viz = Visualizer()
        
        '''
        Sentiment Classification with Naive Bayes
        '''
        nb = Naive_Bayes()
        dfc = DF_Cleaner()
        # pos_df, neg_df, neutral_df, irr_df = dfc.get_sentiment_df(twitter)
        # balanced_df = dfc.balance_df([neg_df, neutral_df, irr_df], neg_df)
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


    def run_doc2vec_logreg(self):
        print("Running Logistic Regression Classification with Doc2Vec")
        twitter = pd.read_csv('data/full-corpus.csv', encoding='utf-8')
        dfc = DF_Cleaner()
        viz = Visualizer()
        d2v = My_Doc2Vec()

        # Balancing and Train Test Split
        pos_df, neg_df, neutral_df, irr_df = dfc.get_sentiment_df(twitter)
        balanced_df = dfc.balance_df([neg_df, neutral_df, irr_df], neg_df)
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

        d2v.pickle_model(logreg)
        viz.plot_confusion_matrix(y_test, y_pred, classes=['positive', 'negative', 'neutral', 'irrelevant'], \
                                  title='Logestic Regression with Doc2Vec')
        plt.savefig('media/confusion_matrix/d2v_logreg_confmat.png')
        plt.close()

    def run_doc2vec_naivebayes(self):
        print("Running Naive Bayes Classification with Doc2Vec")
        twitter = pd.read_csv('data/full-corpus.csv', encoding='utf-8')
        dfc = DF_Cleaner()
        viz = Visualizer()

        # Balancing and Train Test Split
        pos_df, neg_df, neutral_df, irr_df = dfc.get_sentiment_df(twitter)
        balanced_df = dfc.balance_df([neg_df, neutral_df, irr_df], neg_df)
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
        # logreg = LogisticRegression(n_jobs=1, C=1e5)
        # logreg.fit(X_train, y_train)
        # y_pred = logreg.predict(X_test)

        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
        print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

        d2v.pickle_model(clf)
        viz.plot_confusion_matrix(y_test, y_pred, classes=['positive', 'negative', 'neutral', 'irrelevant'], \
                                  title='Guassian Navie Bayes with Doc2Vec')
        plt.savefig('media/confusion_matrix/d2v_nb_confmat.png')
        plt.close()

