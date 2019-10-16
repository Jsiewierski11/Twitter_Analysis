
#import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from scipy.stats import entropy

import pandas as pd 
import numpy as np


# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

class Visualizer(object):

    def __init__(self):
        pass


    def plot_coherence(self, model_list, c_v_vals, u_mass_vals, start=2, stop=30, step=3, filepath='media/coherence_viz.png'):
        # stop += 1
        # (model_list, c_v_vals, u_mass_vals) = self.compute_coherence_values(texts=self.corpus,
        #                                                                             start=start,
        #                                                                             stop=stop,
        #                                                                             step=step)

        # Show graph
        x = range(start, stop, step)
        plt.plot(x, c_v_vals, color='blue')
        plt.plot(x, u_mass_vals, color='red')
        plt.xlabel("Number of Topics", fontsize=14)
        plt.ylabel("Coherence score", fontsize=14)
        plt.title("Coherence score using c_v and u_mass Metrics vs Number of Topics")
        plt.legend((c_v_vals, u_mass_vals), ('c_v', 'u_mass'))
        plt.savefig(filepath)
        plt.close()


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
        plt.close()


    def make_pyLDAvis(self, model, bow, id2word, filepath='media/LDA_topics.html'):
        '''
        Saves a pyLDAvis visualization to the media file
        '''
        vis = pyLDAvis.gensim.prepare(model, bow, id2word, mds='mmds')
        pyLDAvis.save_html(vis, filepath)


    '''
    Protected Methods
    '''

    def _sort_wc(self, wc_dict):
        wc_df = pd.DataFrame.from_dict(wc_dict, orient='index')
        return wc_df.sort_values(by=0, ascending=False)