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

from ovmsclient import make_grpc_client

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

def prepare(img_path):
    image = Image.open(img_path)
    img_data = transform_pipeline(image).unsqueeze(0)
    data = np.array(img_data)
    return data


def main_single(img_path, server_url):
    client = make_grpc_client(server_url)
    img_data = prepare(img_path)
    inputs = {"input.1": img_data}
    results = client.predict(inputs=inputs, model_name="vgg16")
    index = np.argmax(results)
    print("Result label index is {}".format(index))


def main_stress(folder, server_url, threads):
    def do_request(x):
        client = make_grpc_client(server_url)
        inputs = {"input.1": x}
        results = client.predict(inputs=inputs, model_name="vgg16")
        return np.argmax(results)

    imgs = os.listdir(folder)

    batch = 100
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
    parser.add_argument("--url", help="url to server", default="localhost:9000")
    parser.add_argument("--threads", help="number of threads", default="2")
    args = parser.parse_args()

    if args.mode == "single":
        main_single(args.img, args.url)
    elif args.mode == "stress":
        main_stress(args.img, args.url, int(args.threads))
    else:
        print("Unexpected mode {}".format(args.mode))
