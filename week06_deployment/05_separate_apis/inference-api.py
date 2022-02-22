import json
import logging

from concurrent import futures

import grpc
import numpy as np
import torch
import inference_pb2
import inference_pb2_grpc


class InferenceClassifier(inference_pb2_grpc.ImageClassifierServicer):
    def __init__(self):
        self.model = torch.jit.load('vgg16.pt')
        with open('labels.json', 'r') as f:
            labels_raw = json.loads(f.read())
            self.labels = {int(index): value for index, value in enumerate(labels_raw)}

    def Predict(self, request, context):
        shape = request.shape
        data = np.array(request.data).reshape(*shape)
        features = torch.from_numpy(data).float()
        result = self.model(features).data.numpy().argmax()
        label = self.labels[result]
        return inference_pb2.ImageClassifierOutput(label=label)


def serve():
    # to use processes - https://github.com/grpc/grpc/blob/master/examples/python/multiprocessing/server.py
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_ImageClassifierServicer_to_server(InferenceClassifier(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    print("start serving...")
    serve()
