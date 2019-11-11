import os, sys
sys.path.append(os.path.abspath('..'))
from src.runner import Runner


if __name__ == '__main__':
    runner = Runner()

    '''TF-IDF'''
    # runner.run_naive_bayes_sentiment()
    # runner.run_naive_bayes_topic()

    '''Doc2Vec'''
    # runner.run_doc2vec_logreg()
    # runner.run_doc2vec_naivebayes()

    '''EDA Plots of Data'''
    runner.make_plots()