# -- coding: utf-8 --

# $ docker build -t flask-application:latest .
# $ docker run -d -p 5000:5000 flask-application

from flask import Flask, request, jsonify
import numpy as np
import math

from features.distance import centroidVec, cos_sim
from features.translation import transItem
from models._MF import MF
from features.kerasEncoder import imgToVec


# Flask Endpoint 설정
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

flask_host = "0.0.0.0"
flask_port = "5000"




@app.route("/reco/mf", methods=['POST'])
def get_reco_item_mf():
    req_body = request.get_json()
    res_body = []
    mall_list = req_body['mallList']
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
        sorted_idx = np.argsort(-rates)
        mall_list = [mall_list[x] for x in sorted_idx]
        mall_rate_list = [0.0 if math.isnan(i) else i for i in rates.tolist()]
        user_mall_list = []
        for i in range(len(mall_list)):
            user_mall = {'mall': mall_list[i], 'rate': mall_rate_list[i]}
            user_mall_list.append(user_mall)

        user_item = {
            "userId": str(key),
            "userMallList": user_mall_list
        }
        res_body.append(user_item)

    return jsonify(res_body)


@app.route("/feature/title/translation", methods=['POST'])
def get_title_translation():
    print(request.base_url)
    req_body = request.get_json()
    res_body = {
        "engTitle": transItem(req_body['title'])
    }
    return jsonify(res_body)

@app.route("/feature/image", methods=['POST'])
def get_image_feature():
    req_body = request.get_json()
    res_body = {
        "imageVec": imgToVec(req_body['imageUrl'])
    }
    # res_body = []
    # for d in req_body:
    #     res_body.append({
    #         "id": d['id'],
    #         "mallNm": d['mallNm'],
    #         'imageUrl':d['imageUrl'],
    #         'imageVev':imgToVec(d['imageUrl'])
    #     })
    return jsonify(res_body)

@app.route("/feature/distance", methods=['POST'])
def get_feature_distance():
    req_body = request.get_json()
    recent_item_image_vec_list = list(map(lambda x: x['imageVec'], req_body['recentItemList']))
    user_item_list = req_body['userItemList']
    represent_vec = centroidVec(recent_item_image_vec_list)

    for user_item in user_item_list:
        item = user_item['item']
        item_image_vec = item['imageVec']
        user_item['rate'] = cos_sim(item_image_vec, represent_vec)

    item_rate_list = list(map(lambda i: i['rate'], user_item_list))
    for i in range(len(user_item_list)):
        if math.isnan(item_rate_list[i]):
            user_item_list[i]['rate'] = 0.0
        else:
            user_item_list[i]['rate'] = item_rate_list[i]

    res_body = req_body
    res_body['userItemList'] = user_item_list

    return jsonify(res_body)

if __name__ == "__main__":
    # Run Flask Server
    app.run(host=flask_host, port=flask_port)