from flask import Flask, request, render_template
import json
import requests
import socket

app = Flask(__name__)
PORT = 8080


@app.route('/')
def root():
    return render_template('home.html')


if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)