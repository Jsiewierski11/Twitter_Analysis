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
        self.bow = None
        self.id2word = None
        self.doc_dist = []
        self.doc_loads = []


    def create_corpus(self, corpus):
        self.corpus = corpus
        

    def tokenize_corpus(self, custom_stopwords=False):
        self.corpus = [self._preprocess(x, custom_stopwords=custom_stopwords) for x in self.corpus]

    
    def create_bow(self):
       #create dictionary
        self.id2word = gensim.corpora.Dictionary(self.corpus)
        #create corpus
        texts = self.corpus
        #Term Document Frequency
        self.bow = [self.id2word.doc2bow(text) for text in texts]


    def print_tdf(self):
        # Human readable format of corpus (term-frequency)
        print([[(self.id2word[id], freq) for id, freq in cp] for cp in self.bow])


    def create_lda_model(self, num_topics=4, passes=5, workers=2):
        return gensim.models.LdaMulticore(self.bow, num_topics=num_topics, per_word_topics=True, id2word=self.id2word, passes=passes, workers=passes)

    
    def print_top_words(self, lda_model):
        # Print the Keyword in the 10 topics
        pprint(lda_model.print_topics())


    def print_perplexity_coherence(self, lda_model):
        # Compute Perplexity
        print('\nPerplexity: ', lda_model.log_perplexity(self.bow))  # a measure of how good the model is. lower the better.
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


    def document_topic_distribution(self, lda_model):
        '''
        Returns a list of how much a document loads onto each latent topic
        e.g. - [0.5, 0.2, 0.9, 0.4]
        '''
        doc_tops = lda_model.get_document_topics(self.bow)
        for doc_top in doc_tops:
            self.doc_dist.append([tup[1] for tup in doc_top])
            self.doc_loads.append([tup[0] for tup in doc_top])


    def determine_doc_topic(self, full_doc_list, doc_ind):
        topic_of_doc = []
        for doc in self.doc_dist:
            topic_of_doc.append(np.argmax(doc))

        print(f'document:\n{full_doc_list[doc_ind]}')
        print(f'Is apart of latent topic {topic_of_doc[doc_ind]} with a load value of {self.doc_dist[doc_ind][topic_of_doc[doc_ind]]}')


    def compute_coherence_values(self, start=2, stop=30, step=3):
        """
        Compute c_v coherence for various number of topics
        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics
        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with
                        respective number of topics
        """
        coherence_values = []
        u_mass_vals = []
        model_list = []

        id2word = self.id2word
        # corpus = self.corpus

        for num_topics in range(start, stop, step):
            print('Calculating {}-topic model'.format(num_topics))
            model = gensim.models.LdaMulticore(self.bow, num_topics=4, id2word=id2word, passes=5, workers=2)

            model_list.append((num_topics, model))
            coherencemodel = CoherenceModel(model=model,
                                            texts=self.corpus,
                                            corpus=self.bow,
                                            dictionary=id2word,
                                            coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

            # u_mass = CoherenceModel(model=model,
            #                                 texts=self.corpus,
            #                                 corpus=self.bow,
            #                                 dictionary=id2word,
            #                                 coherence='u_mass')
            # u_mass_vals.append(u_mass.get_coherence())

        return model_list, coherence_values#, u_mass_vals
    

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
            spanish = self._get_spanish_stopwords()
            custom = self._get_custom_stopwords()
            stopwords.update(spanish)
            # stopwords.update(['http', 'fuck', 'rt'])
            stopwords.update(custom)
        else:
            stopwords = STOPWORDS.copy()

        for token in gensim.utils.simple_preprocess(text, min_len=min_len, max_len=max_len):
            if token not in stopwords:
                result.append(self._lemmatize_stemming(token))
        return result


    def _get_spanish_stopwords(self):
        x = [line.rstrip() for line in open('src/stop_words/spanish.txt')]
        return set(x)


    def _get_custom_stopwords(self):
        x = [line.rstrip() for line in open('src/stop_words/custom.txt')]
        return set(x)
    
    