# Mathematical/data cleaning tools
import numpy as np
import pandas as pd

#gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.utils import to_unicode
from gensim.parsing.preprocessing import STOPWORDS
import gensim.corpora as corpora
from gensim.models import CoherenceModel

#nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk


class Cleaner(object):
    
    def __init__(self, corpus=None):
        self.corpus = corpus


    def create_corpus(self, corpus):
        self.corpus = corpus
        

    def tokenize_corpus(self):
        # for i, x in enumerate(self.corpus):
        #     print(i)
        #     print(f'from text in tokenize_corpus: {x}')
        #     x = to_unicode(x, 'UTF-8')
        #     self.corpus[i] = self.preprocess(x)
        self.corpus = [self.preprocess(x) for x in self.corpus]


    def lemmatize_stemming(self, text):
        stemmer = SnowballStemmer('english')
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


    def preprocess(self, text,  min_len=0, max_len=240):
        result = []
        # print(f'from text in preprocess: {text}')
        for token in gensim.utils.simple_preprocess(text, min_len=min_len, max_len=max_len):
            # print(f'token: {token}')
            if token not in gensim.parsing.preprocessing.STOPWORDS:
                result.append(self.lemmatize_stemming(token))
        return result