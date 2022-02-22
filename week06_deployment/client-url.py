import argparse
import os

import concurrent.futures
import random
from collections import Counter

import requests
from furl import furl
import torchvision.transforms as transforms
from PIL import Image


def main_single(img_path, server_url):
    predict_url = str(furl(server_url) / "predict")
    img_url = str(furl("http://image-server:9090") / img_path)
    r = requests.post(predict_url, json={'image_url': img_url})
    print("It is {}".format(r.json()['label']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="path to img")
    parser.add_argument("--url", help="url to server", default="http://localhost:8080")
    args = parser.parse_args()

    main_single(args.img, args.url)
