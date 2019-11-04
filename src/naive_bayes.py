import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Sklearn
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, classification_report


class Naive_Bayes(object):
    def __init__(self):
        self.count_vect = None
        self.tfidf_transformer = None
        self.clf = None


    def balance_df(self, dfs_to_balance, minority_df):
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


    def compute_tf_and_tfidf(self, train_text):
        # Creating tf vector
        self.count_vect = CountVectorizer(stop_words='english')
        X_train_counts = self.count_vect.fit_transform(train_text)

        # Creating tf-idf vector
        self.tfidf_transformer = TfidfTransformer()
        X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)

        return X_train_counts, X_train_tfidf


    def classify(self, X_train_tfidf, y_train, test_text):
        self.clf = MultinomialNB()
        self.clf.fit(X_train_tfidf, y_train)
        y_pred = self.clf.predict(self.count_vect.transform(test_text))
        return y_pred


    def print_metrics(self, y_test, y_pred):
        print(classification_report(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))


    def pickle_model(self, filepath='models/naive_bayes.pkl'):
        # Saving File
        with open(filepath, 'wb') as f:
            pickle.dump(self.clf, f)