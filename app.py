# -- coding: utf-8 --

# $ docker build -t flask-application:latest .
# $ docker run -d -p 5000:5000 flask-application

from flask import Flask, request, jsonify

# Flask Endpoint 설정
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

flask_host = "0.0.0.0"
flask_port = "5000"


@app.route("/test", methods=['GET'])
def test():
    return ""


if __name__ == "__main__":
    # Run Flask Server
    app.run(host=flask_host, port=flask_port, debug=True)
