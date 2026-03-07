import time


METRIC_SHOW_PERIOD = 3.0

class MetricManager:
    def __init__(self, enable_metrics: bool = False):

        self.enable_metrics = enable_metrics
        self.waiting_queue_num = 0
        self.active_requests_num = 0

        self.throughput_tokens_per_second = 0.0
        self.ttft_ms = 0.0
        self.tpot_ms = 0.0
        self.rps = 0.0
        self.time = time.time()
    
    def calculate_throughtput_tokens_per_second(self, tokens_num: int, time_s: float):
        # TODO: Implement throughput calculation
        raise NotImplementedError("Implement throughput calculation")

    def update_waiting_queue_num(self, num: int):
        # TODO: Implement waiting queue number update
        raise NotImplementedError("Implement waiting queue number update")

    def update_active_requests_num(self, num: int):
        # TODO: Implement active requests number update
        raise NotImplementedError("Implement active requests number update")

    def set_no_work(self):
        # TODO: Implement no work state update
        raise NotImplementedError("Implement no work state update")

    def show_metrics(self, stage: str):
        metrix_output = f"""
{stage}
- Throughput tokens per second: {self.throughput_tokens_per_second:.3f}
- TTFT: {self.ttft_ms:.3f} ms
- TPOT: {self.tpot_ms:.3f} ms
- RPS: {self.rps:.3f}
- Waiting queue number: {self.waiting_queue_num}
- Active requests number: {self.active_requests_num}"""
        print("-" * 20 + metrix_output + "\n" + "-" * 20)
