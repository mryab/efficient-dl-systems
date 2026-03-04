import asyncio
import concurrent.futures

import aiohttp
import grpc
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torchvision.transforms as transforms
from PIL import Image
import io
from functools import reduce
import operator
import inference_pb2_grpc
import inference_pb2

executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

transform_pipeline = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


def prepare(img_data):
    image = Image.open(io.BytesIO(img_data))
    img_data = transform_pipeline(image).unsqueeze(0)
    img_shape = list(img_data.shape)
    flat_shape = reduce(operator.mul, img_shape, 1)

    data = img_data.numpy().reshape(flat_shape).tolist()
    return data, img_shape


class ImageRequest(BaseModel):
    image_url: str


class LabelResponse(BaseModel):
    label: str


app = FastAPI()

""""
{
    "image_url": ""...""
}

"""
@app.post("/predict")
async def hello_world(req: ImageRequest):
    async with aiohttp.ClientSession() as session:
        async with session.get(req.image_url) as resp:
            data = await resp.read()
            loop = asyncio.get_event_loop()
            image_data, shape = await loop.run_in_executor(executor, prepare, data)

    async with grpc.aio.insecure_channel('inference-api:50051') as channel:
        service = inference_pb2_grpc.ImageClassifierStub(channel)
        r = await service.Predict(inference_pb2.ImageClassifierInput(
            data=image_data,
            shape=shape
        ))

    label = r.label
    return LabelResponse(label=label)


if __name__ == '__main__':
    uvicorn.run("client-api:app", port=80, host='0.0.0.0')
