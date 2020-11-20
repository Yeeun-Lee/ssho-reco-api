# -- coding: utf-8 --

# $ docker build -t flask-application:latest .
# $ docker run -d -p 5000:5000 flask-application

from flask import Flask, request, jsonify
import numpy as np
from models._MF import MF

# Flask Endpoint 설정
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

flask_host = "0.0.0.0"
flask_port = "5000"

@app.route("/reco/mf", methods=['POST'])
def get_reco_item_mf():
    req_body = request.get_json()
    res_body = []
    item_id_list = req_body['itemIdList']
    user_swipe_score_list = req_body['userSwipeScoreList']

    dict_rate = {}

    for user_swipe_score in user_swipe_score_list:
        key = int(user_swipe_score['userId'])
        value = user_swipe_score['scoreList']
        dict_rate[key] = value

    # P, Q is (7 X k), (k X 5) matrix
    factorizer = MF(dict_rate, latent=3, lr=0.01, reg_param=0.01, epochs=300, verbose=True)
    factorizer.fit()
    factorizer.print_results()
    for key, rates in factorizer.estimated():
        sorted_idx = np.argsort(rates)
        user_item = {"userId": str(key), "itemIdList": [item_id_list[x] for x in sorted_idx]}
        res_body.append(user_item)

    return jsonify(res_body)

if __name__ == "__main__":
    # Run Flask Server
    app.run(host=flask_host, port=flask_port)
