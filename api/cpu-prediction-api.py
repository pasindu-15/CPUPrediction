from flask import Flask, jsonify, request

import sys

import svr_regression

app = Flask(__name__)

# @app.route('/cpu-prediction/<double:req_count>/<double:res_time>/<double:act_con_count>/<double:new_con_count>', methods=['GET'])
@app.route('/cpu-prediction/', methods=['GET'])
def get_cpu_prediction():
    # v = svr_regression.get_predicted_cpu([3827.0, 0.010416260256075255, 193.0, 139.0])
    req_count = float(request.args.get('req'))
    res_time = float(request.args.get('res'))
    act_con_count = float(request.args.get('act'))
    new_con_count = float(request.args.get('new'))
    v = svr_regression.get_predicted_cpu([req_count, res_time, act_con_count, new_con_count])
    response = {"prediction": v[0][0]}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
