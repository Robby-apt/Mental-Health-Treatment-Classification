import numpy as np
from flask import Flask,request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    # for rendering results on index.html
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    # output = round(prediction[0], 2)
    output = prediction[0]

    return render_template('results.html', prediction_text = 'Should you seek treatment? $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)