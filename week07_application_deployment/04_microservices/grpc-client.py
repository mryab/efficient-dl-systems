import argparse
import asyncio
import io
import operator
from functools import reduce

import grpc
import torchvision.transforms as transforms
from PIL import Image
import inference_pb2_grpc
import inference_pb2

transform_pipeline = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


def prepare(img_data):
    image = Image.open(img_data)
    img_data = transform_pipeline(image).unsqueeze(0)
    img_shape = list(img_data.shape)
    flat_shape = reduce(operator.mul, img_shape, 1)

    data = img_data.numpy().reshape(flat_shape).tolist()
    return data, img_shape


def main_single(img_path, server_url):
    image_data, shape = prepare(img_path)

    with grpc.insecure_channel(server_url) as channel:
        service = inference_pb2_grpc.ImageClassifierStub(channel)
        r = service.Predict(inference_pb2.ImageClassifierInput(
            data=image_data,
            shape=shape
        ))
        print("It is {}".format(r.label))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="mode to run")
    parser.add_argument("img", help="path to img")
    parser.add_argument("--url", help="url to server", default="localhost:50051")
    args = parser.parse_args()

    if args.mode == "single":
        main_single(args.img, args.url)
    else:
        print("Unexpected mode {}".format(args.mode))
