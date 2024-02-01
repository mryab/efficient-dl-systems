"""
Source: https://github.com/ternaus/imread_benchmark/blob/master/imread_benchmark/benchmark.py
Benchmark libraries: cv2, skimage, PIL, jpeg4py, imageio

for the case jpeg images => numpy array for RGB image
"""

import argparse
import math
import random
import sys
from abc import ABC
from collections import defaultdict
from pathlib import Path
from timeit import Timer
from typing import Union

import cv2
import imageio
import jpeg4py
import numpy as np
import pandas as pd
import pkg_resources
import skimage
from PIL import Image
from tqdm import tqdm

from torchvision.io import read_image

def print_package_versions():
    packages = ["opencv-python", "pillow-simd", "jpeg4py", "scikit-image", "imageio"]
    package_versions = {"python": sys.version}
    for package in packages:
        try:
            package_versions[package] = pkg_resources.get_distribution(package).version
        except pkg_resources.DistributionNotFound:
            pass
    print(package_versions)


def format_results(images_per_second_for_read, show_std=False):
    if images_per_second_for_read is None:
        return "-"
    result = str(math.floor(np.mean(images_per_second_for_read)))
    if show_std:
        result += " ± {}".format(math.ceil(np.std(images_per_second_for_read)))
    return result


class BenchmarkTest(ABC):
    def __str__(self):
        return self.__class__.__name__

    def run(self, library, image_paths: list):
        operation = getattr(self, library)
        for image in image_paths:
            operation(image)


class GetArray(BenchmarkTest):
    def PIL(self, image_path: str) -> np.array:
        img = Image.open(image_path)
        img = img.convert("RGB")
        return np.asarray(img)

    def opencv(self, image_path: str) -> np.array:
        img = cv2.imread(image_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def jpeg4py(self, image_path: str) -> np.array:
        return jpeg4py.JPEG(image_path).decode()

    def skimage(self, image_path: str) -> np.array:
        return skimage.io.imread(image_path, plugin="matplotlib")

    def imageio(self, image_path: str) -> np.array:
        return imageio.imread(image_path)

    def torch(self, image_path: str) -> np.array:
        return read_image(image_path)


def benchmark(libraries: list, benchmarks: list, image_paths: list, num_runs: int, shuffle: bool) -> defaultdict:
    images_per_second = defaultdict(dict)
    num_images = len(image_paths)

    for library in libraries:
        pbar = tqdm(total=len(benchmarks))
        for benchmark in benchmarks:
            pbar.set_description("Current benchmark: {} | {}".format(library, benchmark))
            if shuffle:
                random.shuffle(image_paths)
            timer = Timer(lambda: benchmark.run(library, image_paths))
            run_times = timer.repeat(number=1, repeat=num_runs)
            benchmark_images_per_second = [1 / (run_time / num_images) for run_time in run_times]
            images_per_second[library][str(benchmark)] = benchmark_images_per_second
            pbar.update(1)

        pbar.close()

    return images_per_second


def parse_args():
    parser = argparse.ArgumentParser(description="Image reading libraries performance benchmark")
    parser.add_argument("-d", "--data-dir", metavar="DIR", help="path to a directory with images")
    parser.add_argument(
        "-i",
        "--num_images",
        default=2000,
        type=int,
        metavar="N",
        help="number of images for benchmarking (default: 2000)",
    )
    parser.add_argument(
        "-r", "--num_runs", default=5, type=int, metavar="N", help="number of runs for each benchmark (default: 5)"
    )
    parser.add_argument(
        "--show-std", dest="show_std", action="store_true", help="show standard deviation for benchmark runs"
    )
    parser.add_argument("-p", "--print-package-versions", action="store_true", help="print versions of packages")
    parser.add_argument("-s", "--shuffle", action="store_true", help="Shuffle the list of images.")
    return parser.parse_args()


def get_image_paths(data_dir: Union[str, Path], num_images: int) -> list:
    image_paths = sorted(Path(data_dir).glob("*.*"))
    return [str(x) for x in image_paths[:num_images]]


def main():
    args = parse_args()
    if args.print_package_versions:
        print_package_versions()

    benchmarks = [GetArray()]

    libraries = ["opencv", "PIL", "jpeg4py", "skimage", "imageio", "torch"]

    image_paths = get_image_paths(args.data_dir, args.num_images)

    images_per_second = benchmark(libraries, benchmarks, image_paths, args.num_runs, args.shuffle)

    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(images_per_second)
    df = df.applymap(lambda r: format_results(r, args.show_std))
    df = df[libraries]

    print(df)


if __name__ == "__main__":
    main()
