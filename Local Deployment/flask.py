"""
A first simple Cloud Foundry Flask app

Author: Ian Huston
License: See LICENSE.txt

"""
from flask import Flask
import os
from flask import Flask, json
from flask import request
from sklearn.externals import joblib
import os

import numpy as np

app = Flask(__name__)

# Get port from environment variable or choose 9099 as local default
port = int(os.getenv("PORT", 9099))

@app.route('/')
def hello_world():
    #return 'Hello World! I am instance ' + str(os.getenv("CF_INSTANCE_INDEX", 0))
	return 'Hello World!'

@app.route('/predict', methods=['GET', 'POST'])
def model_predict():
	if request.method == 'GET':
		return 'Hello World!'
	if request.method == 'POST':
		model = joblib.load('model.pkl')
		jsonObject = json.loads(request.data.decode('utf-8'))
		data = jsonObject['data']
		pred_class = model.predict(data)
		return str(pred_class)
		
'''	if request.method == 'GET':
		name = request.args.get("name")
		return 'Hello ' + name
	if request.method == 'POST':
		model = joblib.load('model.pkl')

		print("*******************")

		jsonObject = json.loads(request.data.decode('utf-8'))
		print(jsonObject)
		print("$$$$$$$$$")
		print(type(jsonObject))
		ndarray = jsonObject['data']
		print("@@@@@@@@@@@@@@@")
		print(ndarray)
		s = model.predict(ndarray)
		return str(s)	
'''		
		
if __name__ == '__main__':
    # Run the app, listening on all IPs with our chosen port number
    app.run(host='127.0.0.1', port=port)
