import json
import sys

import torch
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter


app = Flask(__name__, static_url_path="")
metrics = PrometheusMetrics(app)
PREDICTION_COUNT = Counter("predictions_total", "Number of predictions", ["label"])

model = torch.jit.load('vgg16.pt')
with open('labels.json', 'r') as f:
    labels_raw = json.loads(f.read())
    labels = {int(index): value for index, value in enumerate(labels_raw)}


@app.route("/predict", methods=['POST'])
@metrics.gauge("api_in_progress", "requests in progress")
@metrics.counter("api_invocations_total", "number of invocations")
def predict():
    data = request.get_json(force=True)
    features = torch.tensor(data['data'])

    result = model(features).data.numpy().argmax()
    label = labels[result]

    PREDICTION_COUNT.labels(label=label).inc()

    return jsonify({
        "label": label
    })


if __name__ == '__main__':
    port = int(sys.argv[1])
    app.run(host='0.0.0.0', port=port)
