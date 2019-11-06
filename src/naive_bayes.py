import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Sklearn
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, classification_report


class Naive_Bayes(object):
    def __init__(self):
        self.count_vect = None
        self.tfidf_transformer = None
        self.clf = None


    def compute_tf_and_tfidf(self, train_text, ngram_range=(1, 1)):
        # Creating tf vector
        self.count_vect = CountVectorizer(stop_words='english', ngram_range=ngram_range)
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


    def pickle_model(self, filepath_cv='models/count_vect.pkl', filepath_clf='models/naive_bayes.pkl'):
        # Saving File
        with open(filepath_clf, 'wb') as f:
            pickle.dump(self.clf, f)

        with open(filepath_cv, 'wb') as f:
            pickle.dump(self.count_vect, f)