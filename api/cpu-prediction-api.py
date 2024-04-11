from flask import Flask, jsonify, request

import sys

import svr_regression

app = Flask(__name__)

@app.route('/cpu-prediction', methods=['GET'])
def get_books():
    # v = svr_regression.get_predicted_cpu([3827.0, 0.010416260256075255, 193.0, 139.0])
    v = svr_regression.get_predicted_cpu([3230.0, 0.010447052321981423, 137.0, 129.0])
    response = {"prediction": v[0][0]}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
