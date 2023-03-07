FROM gcc:latest as build

WORKDIR /gtest_build

RUN apt-get update && \
    apt-get install -y cmake

WORKDIR /
RUN wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip && \
    unzip libtorch-shared-with-deps-latest.zip && \
    rm libtorch-shared-with-deps-latest.zip

ADD ./src /app/src

WORKDIR /app/build

RUN cmake -DCMAKE_PREFIX_PATH=/libtorch ../src && \
    cmake --build . --config Release

WORKDIR /app
COPY vgg16.pt .
RUN cp /app/build/vgg_cpp .

ENTRYPOINT ["./vgg_cpp"]
