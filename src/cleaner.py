# Mathematical/data cleaning tools
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

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

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this



class Cleaner(object):
    
    def __init__(self, corpus=None):
        self.corpus = corpus
        self.tdf = None
        self.id2word = None
        self.doc_dist = []
        self.doc_loads = []


    def create_corpus(self, corpus):
        self.corpus = corpus
        

    def tokenize_corpus(self, custom_stopwords=False):
        self.corpus = [self._preprocess(x, custom_stopwords=custom_stopwords) for x in self.corpus]

    
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
        return gensim.models.LdaMulticore(self.tdf, num_topics=num_topics, per_word_topics=True, id2word=self.id2word, passes=passes, workers=passes)

    
    def print_top_words(self, lda_model,):
        # Print the Keyword in the 10 topics
        pprint(lda_model.print_topics())


    def print_perplexity_coherence(self, lda_model):
        # Compute Perplexity
        print('\nPerplexity: ', lda_model.log_perplexity(self.tdf))  # a measure of how good the model is. lower the better.
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=self.corpus, dictionary=self.id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)


    def wc_whole_corpus(self):
        # y = np.array([np.array(xi) for xi in self.corpus])
        # y = np.array(self.corpus)
        y = np.array([y for xi in self.corpus for y in xi])
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))

    
    def plot_wc(self, wc_dict, n=20, filepath='media/tf.png'):
        wc = self._sort_wc(wc_dict)
        wc = wc[:n]
        fig, ax = plt.subplots(figsize=(15, 10))
        plt.bar(wc.index, wc[0], color='g')
        plt.title("Top 10 Most Frequent Words in the Corpus", fontsize=14)
        plt.xlabel('Words', fontsize=14)
        plt.ylabel('Term Frequency', fontsize=14)
        plt.xticks(rotation=90)
        plt.savefig(filepath)


    def document_topic_distribution(self, lda_model):
        '''
        Returns a list of how much a document loads onto each latent topic
        e.g. - [0.5, 0.2, 0.9, 0.4]
        '''
        doc_tops = lda_model.get_document_topics(self.tdf)
        for doc_top in doc_tops:
            self.doc_dist.append([tup[1] for tup in doc_top])
            self.doc_loads.append([tup[0] for tup in doc_top])


    def determine_doc_topic(self, full_doc_list, doc_ind):
        topic_of_doc = []
        for doc in self.doc_dist:
            topic_of_doc.append(np.argmax(doc))

        print(f'document:\n{full_doc_list[doc_ind]}')
        print(f'Is apart of latent topic {topic_of_doc[doc_ind]} with a load value of {self.doc_dist[doc_ind][topic_of_doc[doc_ind]]}')


    def make_pyLDAvis(self, model):
        '''
        Saves a pyLDAvis visualization to the media file
        '''
        vis = pyLDAvis.gensim.prepare(model, self.tdf, self.id2word, mds='mmds')
        pyLDAvis.save_html(vis, 'media/LDA_10_topics.html')


    '''
    Protected Methods (don't use these methods in main.py)
    '''
    
    def _sort_wc(self, wc_dict):
        wc_df = pd.DataFrame.from_dict(wc_dict, orient='index')
        return wc_df.sort_values(by=0, ascending=False)


    def _lemmatize_stemming(self, text):
        stemmer = SnowballStemmer('english')
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


    def _preprocess(self, text,  min_len=2, max_len=240, custom_stopwords=False):
        result = []
        if custom_stopwords:
            stopwords = STOPWORDS.copy()
            stopwords = set(stopwords)
            stopwords.update(["google", "apple", "que", "es", "rt", "twitter", "http", "microsoft", "la", "el", "en"])
        else:
            stopwords = STOPWORDS.copy()

        for token in gensim.utils.simple_preprocess(text, min_len=min_len, max_len=max_len):
            if token not in stopwords:
                result.append(self._lemmatize_stemming(token))
        return result
    
    