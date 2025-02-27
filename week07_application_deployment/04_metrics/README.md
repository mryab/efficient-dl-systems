# Metrics

Put `vgg16.pt` inside this directory

Run 

```bash
docker-compose up --build
```

Visit
* `http://localhost:8080/metrics` - raw metrics from app
* `http://localhost:3000/` or `http://localhost:9090` - login with `admin`/`admin` - grafana to draw metrics

## Further reading

* Prometheus - https://prometheus.io/
* Prometheus & Flask - https://pypi.org/project/prometheus-flask-exporter/
* Grafana - https://grafana.com/
* Telegraf - https://www.influxdata.com/time-series-platform/telegraf/