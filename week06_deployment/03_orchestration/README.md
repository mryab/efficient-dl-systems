# Orchestration

Put `vgg16.pt` inside this directory

Install minikube & kubectl - https://kubernetes.io/ru/docs/tasks/tools/install-minikube/

Up local cluster
```bash
minikube delete
minikube start --vm-driver=virtualbox --memory 8192 --cpus 2
```

Connect to cluster's docker and build image

```bash
eval $(minikube -p minikube docker-env)

docker build -f Dockerfile.production -t vgg16-inference-server:1.0.0 .
```

Deploy application info cluster

```bash
kubectl apply -f kubes/deployment.yaml --record
kubectl apply -f kubes/service.yaml
```

Run `minikube ip` to get IP address of cluster - use it in client for requests

More

* Kubernetes - https://kubernetes.io/ru/docs/concepts/overview/what-is-kubernetes/
