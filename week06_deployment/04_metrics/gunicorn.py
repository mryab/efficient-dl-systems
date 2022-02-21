import multiprocessing

from prometheus_flask_exporter.multiprocess import GunicornInternalPrometheusMetrics


def child_exit(server, worker):
    GunicornInternalPrometheusMetrics.mark_process_dead_on_child_exit(worker.pid)


bind = '0.0.0.0:8080'
workers = 1  # multiprocessing.cpu_count() * 2 + 1
