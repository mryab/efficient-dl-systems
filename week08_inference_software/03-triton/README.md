# How to

- Prepare model via `train_model.py`
- Put model into `models-repo/vgg16/1`
- Run `docker-compose up`
- Run `python client.py single dataset/5.jpeg` to send request to inference server.

More

* Triton - https://github.com/triton-inference-server/server
* Triton scaling on k8s - https://developer.nvidia.com/blog/deploying-nvidia-triton-at-scale-with-mig-and-kubernetes/
