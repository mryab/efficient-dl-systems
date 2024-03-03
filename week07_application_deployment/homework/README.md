# Service deployment

Your task is to create an Instance Segmentation (more specifically, Instance Detection) service.
You can use **any pretrained model** (such as [these](https://pytorch.org/vision/stable/models.html#instance-segmentation), for example). 
Instance segmentation metrics will not be counted in your final grade, but MAP should be above 0.5 in order to pass the tests.

Note that object names are the same as in the [COCO 2017 dataset](https://cocodataset.org/#download). 
You can retrieve them from the [FasterRCNN_ResNet50_FPN_V2](https://pytorch.org/vision/stable/models.html#instance-segmentation) model:
```python
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

print(FasterRCNN_ResNet50_FPN_V2_Weights.meta['categories'])
```

**[4 points] HTTP endpoint:**
Implement a service which can handle the `POST /predict` query on port `8080`.
Request data is a JSON with the following structure:
```json
{
  "url": "<url to image>"
}
```

Response data is also a JSON with the following structure:
```json
{
  "objects": [
    "<object name>",
    "<object name>",
    ...
  ]
}
```

Example:
```bash
curl -XPOST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"url": "http://images.cocodataset.org/val2017/000000001268.jpg"}' 
...
{
  "objects": [
        "bird",
        "boat",
        "boat",
        "person",
        "person",
        "person",
        "person",
        "cell phone",
        "backpack",
        "handbag",
        "boat"
    ]
}
```

**[3 points] Metric endpoint:**
Implement a service which is able to serve its metrics in the [Prometheus format](https://prometheus.io/docs/concepts/data_model/) via `/metrics` on the `8080` port. 
The most important metric for us is `app_http_inference_count`, the number of HTTP endpoint invocations.

Example:
```bash
curl http://localhost:8080/metrics
...
# HELP app_http_inference_count_total Multiprocess metric
# TYPE app_http_inference_count_total counter
app_http_inference_count_total 12.0
```

**[3 points] GRPC endpoint:**
Implement a separate [gRPC](https://grpc.io/) service on the `9090` port. 
See the `inference.proto` file in the `proto` directory. 
The contract is the same as for the HTTP endpoint.

### How to submit?

* Create a private GitHub repository
* Add [solution-extractor59](https://github.com/solution-extractor59) to collaborators. **It may take up to 30 minutes for your request to be processed**.
* Put `Dockerfile` in the root of the repository. This Dockerfile should assemble all code in your repo **and the model checkpoint** into the working service.
* Go to [http://week07.hareburrow.space:8080](http://week07.hareburrow.space:8080) , use `student`/`Student!1` for login.
* Build `week07-pipeline` with your repo as a parameter.
* After successful build, click `Keep this build forever`
* This link to your successful (or maybe only partially successful) is the solution for this homework - send it via anytask/LMS.
* Also, add the link to your GitHub repo and the commit hash which corresponds to your submission.

### Notes

* Pretrained PyTorch model `maskrcnn_resnet50_fpn` with `score_threshold = 0.75` works fine for this task
* You are not limited to running only one process inside docker - use `supervisord` to run as many processes as you want
* You can find actual tests in file `tests.py`
* In order to sanity-check your model, tests check that MAP is greater than 0.5 - see tests for details