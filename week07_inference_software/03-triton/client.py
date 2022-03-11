import argparse
import json
import os

import concurrent.futures
import random
import time
from collections import Counter

import numpy as np
import requests
from furl import furl
import torchvision.transforms as transforms
from PIL import Image

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype


transform_pipeline = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


def prepare(img_path):
    image = Image.open(img_path)
    img_data = transform_pipeline(image).unsqueeze(0)
    data = np.array(img_data)
    return data


def main_single(img_path, server_url):
    triton_client = httpclient.InferenceServerClient(url=server_url, concurrency=1)
    img_data = prepare(img_path)
    img_input = httpclient.InferInput('input__0', img_data.shape, "FP32")
    img_input.set_data_from_numpy(img_data)
    result = triton_client.infer('vgg16', [img_input])
    index = result.as_numpy('output__0')[0].argmax()
    print("Result label index is {}".format(index))


def main_stress(folder, server_url, threads):
    def do_request(x):
        triton_client = httpclient.InferenceServerClient(url=server_url, concurrency=1)
        img_input = httpclient.InferInput('input__0', x.shape, "FP32")
        img_input.set_data_from_numpy(x)
        result = triton_client.infer('vgg16', [img_input])
        return result.as_numpy('output__0')[0].argmax()

    imgs = os.listdir(folder)

    batch = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        while True:
            start = time.time()
            print("preparing")
            imgsd = [prepare(str(furl(folder)/random.choice(imgs))) for _ in range(batch)]
            print("start batch")
            futures = [executor.submit(do_request, img_data) for img_data in imgsd]

            labels = Counter()
            for future in concurrent.futures.as_completed(futures):
                labels[future.result()] += 1
            end = time.time()
            perf = batch / (end - start)
            print("Current batch: {}. Perf = {} img/sec".format(labels, perf))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="mode to run")
    parser.add_argument("img", help="path to img")
    parser.add_argument("--url", help="url to server", default="localhost:8000")
    parser.add_argument("--threads", help="number of threads", default="2")
    args = parser.parse_args()

    if args.mode == "single":
        main_single(args.img, args.url)
    elif args.mode == "stress":
        main_stress(args.img, args.url, int(args.threads))
    else:
        print("Unexpected mode {}".format(args.mode))
