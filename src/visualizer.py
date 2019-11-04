import pandas as pd 
import numpy as np

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, classification_report

class Visualizer(object):

    def __init__(self):
        pass


    def plot_coherence(self, model_list, c_v_vals, start=2, stop=30, step=3, \
                       filepath='media/coherence_viz.png', color='blue', \
                       title="Coherence score using c_v Metrics vs Number of Topics"):

        # Show graph
        x = range(start, stop, step)
        plt.plot(x, c_v_vals, color=color)
        # plt.plot(x, u_mass_vals, color='red')
        plt.xlabel("Number of Topics", fontsize=14)
        plt.ylabel("Coherence score", fontsize=14)
        plt.title(title)
        # plt.legend((c_v_vals, u_mass_vals), ('c_v', 'u_mass'))
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


    def plot_categories_bar(self):
        '''
            - 1142 documents labeled as having a topic of Apple
            - 1317 documents labeled as having a topic of Google
            - 1364 documents labeled as having a topic of Microsoft
            - 1290 documents labeled as having a topic of Twitter
        '''
        topic_values = [1142, 1317, 1364, 1290]
        topic_labels = ['Apple', 'Google', 'Microsoft', 'Twitter']

        fig, ax = plt.subplots(figsize=(10, 10))
        plt.bar(topic_labels, topic_values, color=['grey', 'yellow', 'blue', 'cyan'])
        plt.title('Number of Tweets in Each Predefined Topic', fontsize=14)
        plt.xlabel('Predefined Topics', fontsize=14)
        plt.ylabel('Number of Tweets', fontsize=14)
        plt.savefig('media/categories_bar.png')


    def plot_categories_pie(self):
        '''
            - 1142 documents labeled as having a topic of Apple
            - 1317 documents labeled as having a topic of Google
            - 1364 documents labeled as having a topic of Microsoft
            - 1290 documents labeled as having a topic of Twitter
        '''
        topic_values = [1142, 1317, 1364, 1290]
        topic_labels = ['1142 Tweets', '1317 Tweets', '1364 Tweets', '1290 Tweets']
        legend_labels = ['Apple', 'Google', 'Microsoft', 'Twitter']

        fig, ax = plt.subplots(figsize=(7, 3))
        
        wedges, texts, autotexts = plt.pie(topic_values, \
                                           labels=topic_labels, \
                                           colors=['grey', 'yellow', 'blue', 'cyan'], \
                                           autopct='%1.1f%%', \
                                           textprops=dict(color="black"))
        plt.legend(wedges, legend_labels,
                   title="Topics",
                   loc="center left",
                   bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.title('Number of Tweets in Each Predefined Topic', fontsize=14)
        plt.savefig('media/categories_pie.png')
        # plt.show()


    def count_sentiments(self, df):
        pos_df = df[df['Sentiment'] == 'positive']
        num_pos = len(pos_df)

        neg_df = df[df['Sentiment'] == 'negative']
        num_neg = len(neg_df)

        neutral_df = df[df['Sentiment'] == 'neutral']
        num_neu = len(neutral_df)

        irr_df = df[df['Sentiment'] == 'irrelevant']
        num_irr = len(irr_df)

        return [num_neu, num_pos, num_neg, num_irr]


    def plot_sentiments_pie(self, df=None, title='Number of Tweets for Each Sentiment Topic', filepath='media/sentiment_pie.png'):
        '''
            - 519 documents labeled as having a positive sentiment
            - 572 documents labeled as having a negative sentiment
            - 2333 documents labeled as having a neutral sentiment
            - 1689 documents labeled as having an irrelevant sentiment
        '''
        if df is None:
            sentiment_values = [2333, 519, 572, 1689]
            sentiment_labels = ['2333 Tweets', '519 Tweets', '572 Tweets', '1689 Tweets']
            legend_labels = ['Neutral', 'Postive', 'Negative', 'Irrelevant']

            fig, ax = plt.subplots(figsize=(7, 3))
            
            wedges, texts, autotexts = plt.pie(sentiment_values, \
                                            labels=sentiment_labels, \
                                            colors=['grey', 'green', 'red', 'cyan'], \
                                            autopct='%1.1f%%', \
                                            textprops=dict(color="black"))
            plt.legend(wedges, legend_labels,
                    title="Topics",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1))
            
            plt.title('Number of Tweets for Each Sentiment Topic', fontsize=14)
            plt.savefig('media/sentiment_pie.png')
        else:
            sentiment_values = self.count_sentiments(df)
            sentiment_labels = [f'{sentiment_values[0]} Tweets', \
                                f'{sentiment_values[1]} Tweets', \
                                f'{sentiment_values[2]} Tweets', \
                                f'{sentiment_values[3]} Tweets']
            legend_labels = ['Neutral', 'Postive', 'Negative', 'Irrelevant']

            fig, ax = plt.subplots(figsize=(5, 3))
            
            wedges, texts, autotexts = plt.pie(sentiment_values, \
                                            labels=sentiment_labels, \
                                            colors=['grey', 'green', 'red', 'cyan'], \
                                            autopct='%1.1f%%', \
                                            textprops=dict(color="black"))
            plt.legend(wedges, legend_labels,
                    title="Topics",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1))
            
            plt.title(title, fontsize=14)
            plt.savefig(filepath)


    
    def plot_confusion_matrix(self, y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
    #     classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax
   
    '''
    Protected Methods
    '''

    def _sort_wc(self, wc_dict):
        wc_df = pd.DataFrame.from_dict(wc_dict, orient='index')
        return wc_df.sort_values(by=0, ascending=False)