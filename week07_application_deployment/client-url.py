import argparse

import requests
from furl import furl


def main_single(img_path, server_url):
    if not img_path.startswith('https://'):
        img_path = str(furl("http://image-server:9091") / img_path)

    predict_url = str(furl(server_url) / "predict")
    r = requests.post(predict_url, json={'image_url': img_path})
    print("It is {}".format(r.json()['label']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="path to img")
    parser.add_argument("--url", help="url to server", default="http://localhost:8081")
    args = parser.parse_args()

    main_single(args.img, args.url)
