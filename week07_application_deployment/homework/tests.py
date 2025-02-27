import json
import os
import statistics

import grpc
import requests
from furl import furl
import pytest

import inference_pb2_grpc
import inference_pb2
from prometheus_client.parser import text_string_to_metric_families


@pytest.fixture(scope='session')
def eval_data():
    with open('eval.json', 'r') as f:
        return json.loads(f.read())


@pytest.fixture(scope='session')
def server_ip():
    server_ip_value = os.environ['DOCKER_IP']
    if server_ip_value is None:
        pytest.fail("No cluster ip were provided")
    return server_ip_value


@pytest.fixture(scope='session')
def http_host(server_ip):
    return "http://{}:8080/".format(server_ip)


@pytest.fixture(scope="session")
def grpc_host(server_ip):
    return "{}:9090".format(server_ip)


def get_metric_value(samples):
    if len(samples) == 0:
        return 0
    return samples[0].value


def parse_prom(metrics_data):
    return {
        m.name: get_metric_value(m.samples)
        for m in text_string_to_metric_families(metrics_data)
    }


def get_image_link(image_name):
    return "http://images.cocodataset.org/val2017/{}".format(image_name)


def calc_score(actual, predicted):
    actual_copy = [x for x in actual]
    score = 0
    for label in predicted:
        if label in actual_copy:
            score += 1
            actual_copy.remove(label)
    return 2 * score / (len(actual) + len(predicted))


@pytest.mark.run(order=1)
def test_http_endpoint(http_host, eval_data, capsys):
    with capsys.disabled():
        predict_url = str(furl(http_host) / "predict")
        scores = []
        for img_name, labels in eval_data.items():
            print("Processing {}".format(img_name))
            img_url = get_image_link(img_name)
            r = requests.post(predict_url, json={"url": img_url})
            predicted_labels = r.json()['objects']
            scores.append(calc_score(labels, predicted_labels))

        mean_score = statistics.mean(scores)
        assert mean_score > 0.5


@pytest.mark.run(order=2)
def test_grpc_endpoint(grpc_host, eval_data, capsys):
    with capsys.disabled():
        with grpc.insecure_channel(grpc_host) as channel:
            scores = []
            for img_name, labels in eval_data.items():
                print("Processing {}".format(img_name))
                img_url = get_image_link(img_name)
                service = inference_pb2_grpc.InstanceDetectorStub(channel)
                r = service.Predict(inference_pb2.InstanceDetectorInput(
                    url=img_url
                ))
                predicted_labels = r.objects
                scores.append(calc_score(labels, predicted_labels))

            mean_score = statistics.mean(scores)
            assert mean_score > 0.5


@pytest.mark.run(order=3)
def test_http_metrics(http_host, eval_data):
    predict_url = str(furl(http_host) / "predict")
    metrics_url = str(furl(http_host) / "metrics")
    img_name, _ = next(iter(eval_data.items()))
    img_url = get_image_link(img_name)

    init = parse_prom(requests.get(metrics_url).text)
    print(init)
    init_value = int(init['app_http_inference_count'])

    r = requests.post(predict_url, json={'url': img_url})
    assert r.status_code == 200

    next_ = parse_prom(requests.get(metrics_url).text)
    next_value = int(next_['app_http_inference_count'])

    assert next_value == init_value + 1
