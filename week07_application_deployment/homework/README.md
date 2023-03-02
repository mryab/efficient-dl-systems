# Week 7 home assignment

We want to create a service for Instance Detection on pictures.
We don't focus on model quality right now, so it is a great idea to use pretrained model for this task (such as [these](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection), for example) .

**[4 points] HTTP endpoint**
Service should be able to handle `POST /predict` query on port `8080`. Request data is JSON with following structure
```json
{
  "url": "<url to image>"
}
```

Response data is also JSON with the following structure
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
curl -XPOST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"url": "https://storage.yandexcloud.net/effdl2022-coco/000000001268.jpg"}'
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

**[3 points] Metric endpoint**
Service should be able to serve its metrics in the [prometheus format](https://prometheus.io/docs/concepts/data_model/) via `/metrics` on `8080` port. The most important metric for us is `app_http_inference_count` - the number of http endpoint invocations.

Example:
```bash
curl http://localhost:8080/metrics
...
# HELP app_http_inference_count_total Multiprocess metric
# TYPE app_http_inference_count_total counter
app_http_inference_count_total 12.0
```

**[3 points] GRPC endpoint**
We also want a separate [gRPC](https://grpc.io/) service on `9090` port. See proto files in `protos` directory. The contract is the same as for the HTTP endpoint.

### How to submit?

* Create private github repository
* Add [edu-automation-bot](https://github.com/edu-automation-bot) to collaborators (all invitations will be accepted manually, so don't hesitate to notify @adkosm).
* Put `Dockerfile` in the root of repository. This dockerfile should assemble all code in your repo into the working service.
* Go to [effdl.hareburrow.space](http://effdl.hareburrow.space/) , use `student`/`student123!` for login.
* Build `service-deployment-week-07` pipeline with your repo as a parameter.
* After successful build, click `Хранить эту сборку вечно`
* This link to your successful (or maybe only partially successful) build is the answer for this homework - send it via anytask.

### Notes

* Pretrained pytorch model `maskrcnn_resnet50_fpn` with `score_threshold = 0.75` works fine for this task
* You are not limited to run only one process inside docker - use `supervisord` to run as many processes as you want. See [supervisord](./../supervisord) folder for example.
* You can find actual tests in file `tests.py`
* In order to sanity check your model, tests check that mean IoU is greater than 0.5 - see tests for details
* Instance classes are the standard from torchvision docs for instance segmentation models - https://pytorch.org/vision/master/auto_examples/plot_visualization_utils.html#instance-seg-output
