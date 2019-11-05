from flask import Flask, request, render_template
import json
import requests
import socket
import pickle

app = Flask(__name__)
PORT = 8080


with open('../models/naive_bayes_companies.pkl', 'rb') as f:
    clf_company = pickle.load(f)


with open('../models/naive_bayes_sentiment.pkl', 'rb') as f:
    clf_sentiment = pickle.load(f)


@app.route('/', methods=['GET'])
def root():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the article to be classified from an input form and use the
    model to classify.
    """
    data = str(request.form['tweet_body'])
    pred_company = str(clf_company.predict([[data]]))
    pred_sent = str(clf_sent.predict([[data]]))
    return render_template('predict.html', tweet=data, pred_company=pred_company, pred_sent=pred_sent)


if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)