from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


class Kmeans_Operator(object):

    def __init__(self, n=4):
        self.vectorizer = None
        self.X = None
        self.n = n


    def set_vec_doc_mat(self, documents):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        X = self.vectorizer.fit_transform(documents)
        self.X = X.toarray()

    
    def create_and_fit(self, max_iter=100):
        model = KMeans(n_clusters=self.n, init='k-means++', max_iter=max_iter, n_init=1)
        model.fit(self.X)
        return model


    def print_terms_per_cluster(self, model):
        print("Top terms per cluster:")
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = self.vectorizer.get_feature_names()
        for i in range(self.n):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind]),
            print


    def predict_cluster(self, model, document):
        Y = self.vectorizer.transform(document)
        prediction = model.predict(Y)
        print(prediction)
        return prediction