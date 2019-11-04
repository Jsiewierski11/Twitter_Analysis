import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DF_Cleaner(object):
    def __init__(self):
        pass

    def get_sentiment_corpus(self, df):
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
        apple_df = df[df['Topic'] == 'apple']
        google_df = df[df['Topic'] == 'google']
        ms_df = df[df['Topic'] == 'microsoft']
        twitter_df = df[df['Topic'] == 'twitter']
        return apple_df, google_df, ms_df, twitter_df


    def get_sentiment_df(self, df):
        pos_df = df[df['Sentiment'] == 'positive']
        neg_df = df[df['Sentiment'] == 'negative']
        neutral_df = df[df['Sentiment'] == 'neutral']
        irr_df = df[df['Sentiment'] == 'irrelevant']
        return pos_df, neg_df, neutral_df, irr_df