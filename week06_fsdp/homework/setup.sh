#!/usr/bin/env bash
set -euxo pipefail

sudo rm -rf torchtitan
git clone -q https://github.com/pytorch/torchtitan
git -C torchtitan checkout -q 49c6d6fc15ef644e5c3b1003ad4e0d9ea5fcb9a9
curl -s https://gist.githubusercontent.com/antony-frolov/c2e69bbda2b4418b1ab1c99839c55877/raw/c873709f6fe34dbf8ba678302e4fa92d6ed8c7f1/1b.patch -o 1b.patch
patch -s -p1 -i ../1b.patch -d torchtitan
sudo pip install -q fire triton -r ./torchtitan/requirements.txt ./torchtitan
sudo apt-get update -qq && sudo apt-get install -qq pciutils
