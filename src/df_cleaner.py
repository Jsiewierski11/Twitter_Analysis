import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

#nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk


class DF_Cleaner(object):
    '''This class handles any maniuplations that need to be made to the DataFrame for either EDA or training the model.'''

    def __init__(self):
        pass

    def get_sentiment_corpus(self, df):
        '''
        INPUT: Pandas DataFrame that was wrttien in from the full-corpus.csv file
        OUTPUT: Four different numpy arrays containing the tweet text for each of the 4 different sentiments.
        '''

        pos_df = df[df['Sentiment'] == 'positive']
        neg_df = df[df['Sentiment'] == 'negative']
        neutral_df = df[df['Sentiment'] == 'neutral']
        irr_df = df[df['Sentiment'] == 'irrelevant']
        pos_corpus = pos_df['TweetText'].to_numpy()
        neg_corpus = neg_df['TweetText'].to_numpy()
        neutral_corpus = neutral_df['TweetText'].to_numpy()
        irr_corpus = irr_df['TweetText'].to_numpy()
        return pos_corpus, neg_corpus, neutral_corpus, irr_corpus


    def get_topics_corpus(self, df):
        '''
        INPUT: Pandas DataFrame that was wrttien in from the full-corpus.csv file
        OUTPUT: Four different numpy arrays containing the tweet text for each of the 4 different topics.
        '''

        apple_df = df[df['Topic'] == 'apple']
        google_df = df[df['Topic'] == 'google']
        ms_df = df[df['Topic'] == 'microsoft']
        twitter_df = df[df['Topic'] == 'twitter']
        apple_corpus = apple_df['TweetText'].to_numpy()
        google_corpus = google_df['TweetText'].to_numpy()
        ms_corpus = ms_df['TweetText'].to_numpy()
        twitter_corpus = twitter_df['TweetText'].to_numpy()
        return apple_corpus, google_corpus, ms_corpus, twitter_corpus


    def get_topics_df(self, df):
        '''
        INPUT: Pandas DataFrame that was wrttien in from the full-corpus.csv file
        OUTPUT: Four different Pandas DataFrames containing tweets for each of the 4 different topics.
        '''

        apple_df = df[df['Topic'] == 'apple']
        google_df = df[df['Topic'] == 'google']
        ms_df = df[df['Topic'] == 'microsoft']
        twitter_df = df[df['Topic'] == 'twitter']
        return apple_df, google_df, ms_df, twitter_df


    def get_sentiment_df(self, df):
        '''
        INPUT: Pandas DataFrame that was wrttien in from the full-corpus.csv file
        OUTPUT: Four different Pandas DataFrames containing tweets for each of the 4 different sentiments.
        '''

        pos_df = df[df['Sentiment'] == 'positive']
        neg_df = df[df['Sentiment'] == 'negative']
        neutral_df = df[df['Sentiment'] == 'neutral']
        irr_df = df[df['Sentiment'] == 'irrelevant']
        return pos_df, neg_df, neutral_df, irr_df


    def balance_df(self, dfs_to_balance, minority_df):
        '''
        Function to create a data frame where all classes are undersampled to the minority class.

        INPUT:
            dfs_to_balance (list): List of DataFrames that need to be undersampled (e.g. [DF1, DF2, DF2])
            minority_df (Pandas DataFrame): Pandas Dataframe that each dataframe in the list will be undersampled to match.
        OUTPUT: A Pandas DataFrame that has all balanced classes
        '''

        dfs = []
        for df in dfs_to_balance:
            # Downsample majority class to match minority class
            df_down = resample(df,
                                replace=False,    # sample without replacement
                                n_samples=len(minority_df),     # to match minority class
                                random_state=42) # reproducible results
            dfs.append(df_down)
        dfs.append(minority_df)
        df_balanced = pd.concat(dfs)
        return df_balanced


    def wc_corpus(self, doc_array):
        '''
        Function to convert numpy array of documents to a dictionary of word counts.

        INPUT: Numpy array of numpy arrays containing text.
        OUTPUT: Python Dictionary of word counts.
        '''

        y = np.array([y for xi in doc_array for y in xi])
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))


    def preprocess(self, text,  min_len=2, max_len=240, remove_common=False):
        '''
        Function to remove stop words and perform lemmatization.

        INPUT:
            - text (str): numpy array of tweet text.
            - min_len (int): words with less characters than the value of min_len will be removed.
            - max_len (int): words with more character than the value of max_len will be removed.
            - remove_common (bool): add common words in the corpus to the stopwords list.
        OUTPUT: cleaned string
        '''

        result = []
        stopwords = STOPWORDS.copy()
        stopwords = set(stopwords)
        spanish = self._get_spanish_stopwords()
        stopwords.update(spanish)
        stopwords.update(['http', 'fuck', 'rt'])
        if remove_common:
            stopwords.update(['google', 'apple', 'twitter', 'microsoft'])

        for token in gensim.utils.simple_preprocess(text, min_len=min_len, max_len=max_len):
            if token not in stopwords:
                result.append(self._lemmatize_stemming(token))
        return result


    def doc_array_to_str(self, doc_array):
        result = ''
        for doc in doc_array: 
            result += ' '.join([word for word in doc]) 
            result += ' '
        result = result[:-1]
        return result


    def _lemmatize_stemming(self, text):
        '''
        Helper function that gets called in preprocess() to lemmatize words.
        INPUT: text is a str of a single word.
        OUTPUT: str of the word after lemmatization.
        '''
        
        stemmer = SnowballStemmer('english')
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


    def _get_spanish_stopwords(self):
        '''
        Function to read in text file of spanish words to add to stopword list.
        INPUT: None
        OUTPUT: a set of str that can easily be added to a stopwords list.
        '''

        x = [line.rstrip() for line in open('stop_words/spanish.txt')]
        return set(x)