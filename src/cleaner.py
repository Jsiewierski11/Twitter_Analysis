# Mathematical/data cleaning tools
import numpy as np
import pandas as pd
from pprint import pprint

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
        self.tdf = None
        self.id2word = None


    def create_corpus(self, corpus):
        self.corpus = corpus
        

    def tokenize_corpus(self):
        # for i, x in enumerate(self.corpus):
        #     print(i)
        #     print(f'from text in tokenize_corpus: {x}')
        #     x = to_unicode(x, 'UTF-8')
        #     self.corpus[i] = self.preprocess(x)
        self.corpus = [self._preprocess(x) for x in self.corpus]

    
    def create_tdf(self):
       #create dictionary
        self.id2word = gensim.corpora.Dictionary(self.corpus)
        #create corpus
        texts = self.corpus
        #Term Document Frequency
        self.tdf = [self.id2word.doc2bow(text) for text in texts]


    def print_tdf(self):
        # Human readable format of corpus (term-frequency)
        print([[(self.id2word[id], freq) for id, freq in cp] for cp in self.tdf])


    def create_lda_model(self, num_topics=4, passes=5, workers=2):
        return gensim.models.LdaMulticore(self.tdf, num_topics=num_topics, id2word=self.id2word, passes=passes, workers=passes)

    
    def print_top_words(self, lda_model,):
        # Print the Keyword in the 10 topics
        pprint(lda_model.print_topics())


    def _lemmatize_stemming(self, text):
        stemmer = SnowballStemmer('english')
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


    def _preprocess(self, text,  min_len=0, max_len=240):
        result = []
        # print(f'from text in preprocess: {text}')
        for token in gensim.utils.simple_preprocess(text, min_len=min_len, max_len=max_len):
            # print(f'token: {token}')
            if token not in gensim.parsing.preprocessing.STOPWORDS:
                result.append(self._lemmatize_stemming(token))
        return result
    
    