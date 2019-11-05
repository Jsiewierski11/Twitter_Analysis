from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from tqdm import tqdm
from sklearn import utils
import pickle

class My_Doc2Vec(object):
    def __init__(self):
        self.model_dbow = None


    def vec_for_learning(self, tagged_docs):
        sents = tagged_docs.values
        targets, regressors = zip(*[(doc.tags[0], self.model_dbow.infer_vector(doc.words, steps=20)) for doc in sents])
        return targets, regressors


    def tokenize_text(self, text):
        tokens = []
        # Adding to stopwords
        stopwords = STOPWORDS.copy()
        stopwords = set(stopwords)
        spanish = self._get_spanish_stopwords()
        stopwords.update(spanish)
        stopwords.update(['http', 'fuck', 'rt'])

        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if word not in stopwords:
                    if len(word) < 2:
                        continue
                    tokens.append(self._lemmatize_stemming(word.lower()))
        return tokens


    def tag_doc(self, test, train):
        test_tagged = test.apply(lambda r: TaggedDocument(words=self.tokenize_text(r['TweetText']), tags=[r['Sentiment']]), axis=1)
        train_tagged = train.apply(lambda r: TaggedDocument(words=self.tokenize_text(r['TweetText']), tags=[r['Sentiment']]), axis=1)
        return test_tagged, train_tagged
        

    def create_model_and_vocab(self, train_tagged, cores = 2):
        # dbow stands for Distributed Bag of Words
        self.model_dbow = Doc2Vec(dm=0, vector_size=30, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
        self.model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])


    def train_model(self, test_tagged, train_tagged, n_epochs=30):
        for epoch in range(n_epochs):
            self.model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
            self.model_dbow.alpha -= 0.002
            self.model_dbow.min_alpha = self.model_dbow.alpha

    
    def pickle_model(self, clf, filepath='models/doc2vec_logreg.pkl'):
        # Saving File
        with open(filepath, 'wb') as f:
            pickle.dump(clf, f)


    '''
    Protected Methods
    Do not use outside of this class!
    '''
    def _get_spanish_stopwords(self):
        x = [line.rstrip() for line in open('src/stop_words/spanish.txt')]
        return set(x)

    def _lemmatize_stemming(self, text):
        stemmer = SnowballStemmer('english')
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))