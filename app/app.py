from flask import Flask, request, render_template
import json
import requests
import socket
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
PORT = 8080
count_vect = CountVectorizer()


# Classifier Models
with open('../models/naive_bayes_companies.pkl', 'rb') as f:
    clf_company = pickle.load(f)

with open('../models/naive_bayes_sent.pkl', 'rb') as f:
    clf_sent = pickle.load(f)


# Count Vectorizer Models
with open('../models/count_vect_companies.pkl', 'rb') as f:
    count_vect_companies = pickle.load(f)

with open('../models/count_vect_sent.pkl', 'rb') as f:
    count_vect_sent = pickle.load(f)


@app.route('/', methods=['GET'])
def root():
    return render_template('home.html')


@app.route('/model_results')
def model_results():
    return render_template('model_results.html')


@app.route('/wordclouds')
def wordclouds():
    return render_template('wordclouds.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Recieve the article to be classified from an input form and use the
    model to classify.
    """
    data = str(request.form['tweet_body'])
    # print(f'Data to predict:\n{data}')
    # print(f'Type of data variable: {type(data)}')
    pred_company = str(clf_company.predict(count_vect_companies.transform([data]))[0])
    pred_sent = str(clf_sent.predict(count_vect_sent.transform([data]))[0])
    return render_template('predict.html', tweet=data, pred_company=pred_company, pred_sent=pred_sent)


if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)