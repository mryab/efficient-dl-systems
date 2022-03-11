# Kubernetes

* Create k8s cluster
* Create node group with "Intel Cascade Lake with NVIDIA® Tesla® V100"
* Install helm, kubectl
* Create vm for fileserver

Install fileserver

```bash
sudo apt-get update
sudo apt-get install nfs-kernel-server

sudo mkdir /data
sudo chown nobody:nogroup /data

sudo nano /etc/exports

/data 10.128.0.0/16(rw,no_subtree_check,fsid=100)
/data 127.0.0.1(rw,no_subtree_check,fsid=100)

sudo systemctl restart nfs-kernel-server
```

Copy model-repo
```bash
scp -r models-repo/* yc-user@IP:/data
```

Set ip of fileserver in charm template

Setup kubectl

```bash
yc managed-kubernetes cluster get-credentials triton-k8s --external
```

Install prometheus 

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install metrics --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false prometheus-community/kube-prometheus-stack
```

Install triton chart
```bash
cd triton-chart
helm dependency build
helm install triton-chart .
# helm upgrade triton-chart .
```

See ports

```bash
kubectl get service triton-chart-traefik
```

Forward metrics & grafana (admin / prom-operator)

```bash
kubectl port-forward service/triton-chart-triton-inference-server 8002:8002
kubectl port-forward service/metrics-grafana 8080:80
```
