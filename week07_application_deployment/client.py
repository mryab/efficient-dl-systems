import argparse
import os

import concurrent.futures
import random
from collections import Counter

import requests
from furl import furl
import torchvision.transforms as transforms
from PIL import Image

transform_pipeline = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


def prepare(img_path):
    image = Image.open(img_path)
    img_data = transform_pipeline(image).unsqueeze(0)
    data = img_data.tolist()
    return data


def main_single(img_path, server_url):
    predict_url = str(furl(server_url) / "predict")
    print("POST {}".format(predict_url))
    data = prepare(img_path)
    r = requests.post(predict_url, json={'data': data})
    print("It is {}".format(r.json()['label']))


def main_stress(folder, server_url, threads):
    predict_url = str(furl(server_url) / "predict")

    do_request = lambda x: requests.post(predict_url, json={'data': x}).json()
    imgs = os.listdir(folder)

    batch = 100
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        while True:
            print("preparing")
            imgsd = [prepare(str(furl(folder)/random.choice(imgs))) for _ in range(batch)]
            print("start batch")
            futures = [executor.submit(do_request, img_data) for img_data in imgsd]

            labels = Counter()
            for future in concurrent.futures.as_completed(futures):
                labels[future.result()['label']] += 1
            print("Current batch: {}".format(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="mode to run")
    parser.add_argument("img", help="path to img")
    parser.add_argument("--url", help="url to server", default="http://localhost:8080")
    parser.add_argument("--threads", help="number of threads", default="2")
    args = parser.parse_args()

    if args.mode == "single":
        main_single(args.img, args.url)
    elif args.mode == "stress":
        main_stress(args.img, args.url, int(args.threads))
    else:
        print("Unexpected mode {}".format(args.mode))
