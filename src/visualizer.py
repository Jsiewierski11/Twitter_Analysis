import pandas as pd 
import numpy as np

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import matplotlib.pyplot as plt

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


    def plot_sentiments_pie(self):
        '''
            - 519 documents labeled as having a positive sentiment
            - 572 documents labeled as having a negative sentiment
            - 2333 documents labeled as having a neutral sentiment
            - 1689 documents labeled as having an irrelevant sentiment
        '''
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
        # plt.show()


    #Define the word cloud function with a max of 200 words
    '''
    Code below is taken from this Kaggle project:
    https://www.kaggle.com/spurryag/beginner-attempt-at-nlp-workflow
    That user sourced the code from here:
    https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc
    '''
    def plot_wordcloud(self, text_lst, num_axs=1, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                    title = None, title_size=40, image_color=False, filepath=None):
        
        # stopwords = set(STOPWORDS)
        # #define additional stop words that are not contained in the dictionary
        # more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown', 'used'}
        # stopwords = stopwords.union(more_stopwords)
        
        
        for i in range(num_axs):

            #Generate the word cloud
            wordcloud = WordCloud(background_color='black',
                            stopwords = stopwords,
                            max_words = max_words,
                            max_font_size = max_font_size, 
                            random_state = 42,
                            width=800, 
                            height=400,
                            mask = mask)
            wordcloud.generate(str(text))


        #set the plot parameters
        plt.figure(figsize=figure_size)
        if image_color:
            image_colors = ImageColorGenerator(mask);
            plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
            plt.title(title, fontdict={'size': title_size,  
                                    'verticalalignment': 'bottom'})
        else:
            plt.imshow(wordcloud);
            plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                    'verticalalignment': 'bottom'})
        plt.axis('off');
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    '''
    Protected Methods
    '''

    def _sort_wc(self, wc_dict):
        wc_df = pd.DataFrame.from_dict(wc_dict, orient='index')
        return wc_df.sort_values(by=0, ascending=False)