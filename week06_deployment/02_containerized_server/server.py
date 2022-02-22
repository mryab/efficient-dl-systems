import json


import torch
from flask import Flask, request, jsonify

app = Flask(__name__, static_url_path="")

model = torch.jit.load('vgg16.pt')
with open('labels.json', 'r') as f:
    labels_raw = json.loads(f.read())
    labels = {int(index): value for index, value in enumerate(labels_raw)}


@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = torch.tensor(data['data'])

    result = model(features).data.numpy().argmax()
    label = labels[result]

    return jsonify({
        "label": label
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
