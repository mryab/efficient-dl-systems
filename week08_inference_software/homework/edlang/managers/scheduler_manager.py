from typing import List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time

import sys
import os

from edlang.entrypoints.engine import Request, InferenceEngine, BatchResult
from edlang.managers.metric_manager import MetricManager


@dataclass
class SchedulerConfig:
    max_batch_size: int = 8 
    max_waiting_requests: int = 100
    prefill_timeout_ms: float = 50.0
    enable_metrics: bool = False


class EDLangScheduler:

    def __init__(
        self,
        engine: InferenceEngine,
        config: Optional[SchedulerConfig] = None,
    ):
        self.engine = engine
        self.config = config or SchedulerConfig()

        self.waiting_queue = deque()
        self.active_requests = []

        self.next_request_id = 0
        self.metrics_manager = MetricManager(enable_metrics=self.config.enable_metrics)

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ):
        request = Request(
            request_id=self.next_request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        self.waiting_queue.append(request)
        self.next_request_id += 1

        self.metrics_manager.update_waiting_queue_num(len(self.waiting_queue))
        
        return request.request_id
    
    def step(self):
        decode_result = None
        prefill_result = None

        # TODO: Implement step method
        # TODO: First decide how many requests to prefill
        # TODO: Then do decode
        # TODO: Update metrics and inner state
        raise NotImplementedError("Implement step method")
    
    def _decode_step(self):        
        active = [req for req in self.active_requests if not req.is_finished]
        
        if not active:
            return None
        
        # TODO: Do decode for all active requests
        raise NotImplementedError("Implement decode step")
    
    def _prefill_step(self):
        if not self.waiting_queue:
            return None
        
        # Do prefill for some (which?) number of requests
        raise NotImplementedError("Implement prefill step")
    
    def _decide_prefill_batch_size(self):
        # The most simple policy: prefill only if there are no active requests
        num_active = len([r for r in self.active_requests if not r.is_finished])
        
        if num_active > 0:
            return 0
        else:
            return 1
    
    def get_finished_requests(self) -> List[Request]:
        finished = [req for req in self.active_requests if req.is_finished]
        self.active_requests = [req for req in self.active_requests if not req.is_finished]
        return finished
    
    def get_metric_manager(self):
        return self.metrics_manager

    def clear(self):
        self.waiting_queue = deque()
        self.active_requests = []
