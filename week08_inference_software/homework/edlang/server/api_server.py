import time

from dataclasses import dataclass
from typing import List, Optional
from multiprocessing import Process, Queue, Event
import asyncio
import threading
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from edlang.entrypoints.config import EngineConfig
from edlang.entrypoints.engine import InferenceEngine, Request
from edlang.managers.scheduler_manager import EDLangScheduler, SchedulerConfig


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50


class GenerateResponse(BaseModel):
    request_id: int
    generated_text: str
    generated_tokens: List[int]


@dataclass
class RequestMessage:
    request_id: int
    prompt: str
    max_new_tokens: int

@dataclass  
class ResponseMessage:
    request_id: int
    generated_text: str
    generated_tokens: List[int]


app = FastAPI(title="MiniEdlang API Server", version="0.0.1")

class Server:
    def __init__(self, engine_config: EngineConfig, scheduler_config: SchedulerConfig):
        self.engine_config = engine_config
        self.scheduler_config = scheduler_config

        self.request_queue: Queue = Queue()  # API -> Scheduler
        self.response_queue: Queue = Queue()  # Scheduler -> API
        
        self.stop_event = Event()
        self.scheduler_ready_event = Event()
        
        self.pending_requests = {}
        
        self.request_id_counter = 0
        self.request_id_lock = threading.Lock()
        
        self.scheduler_process: Optional[Process] = None


    def _scheduler_worker(self, engine_config: EngineConfig, scheduler_config: SchedulerConfig, request_queue: Queue, 
                         response_queue: Queue, stop_event: Event, scheduler_ready_event: Event):

        engine = InferenceEngine(engine_config)
        scheduler = EDLangScheduler(engine, scheduler_config)

        scheduler_ready_event.set()

        last_time = time.time()
        while True:
            if stop_event.is_set():
                print("Scheduler worker: stop_event received, shutting down...")
                break
            
            try:
                while not request_queue.empty():
                    request_msg = request_queue.get_nowait()
                    scheduler.add_request(prompt=request_msg.prompt, max_new_tokens=request_msg.max_new_tokens)
            except:
                pass
            
            if stop_event.is_set():
                print("Scheduler worker: stop_event received before step, shutting down...")
                break
            
            scheduler.step()

            if stop_event.is_set():
                print("Scheduler worker: stop_event received after step, shutting down...")
                break
            
            finished_requests = scheduler.get_finished_requests()

            for request in finished_requests:
                generated_text = engine.get_generated_text(request)
                response = ResponseMessage(
                    request_id=request.request_id,
                    generated_text=generated_text,
                    generated_tokens=request.generated_tokens or [],
                )
                response_queue.put(response)
            
            time.sleep(0.001)

    def start_scheduler(self):
        if self.scheduler_process is not None and self.scheduler_process.is_alive():
            return
        
        self.scheduler_process = Process(
            target=self._scheduler_worker,
            args=(
                self.engine_config,
                self.scheduler_config,
                self.request_queue,
                self.response_queue,
                self.stop_event,
                self.scheduler_ready_event
            ),
            daemon=True,
        )
        self.scheduler_process.start()
        print(f"Scheduler process started with PID {self.scheduler_process.pid}")
        self.scheduler_ready_event.wait()

    def stop_scheduler(self):
        if self.scheduler_process is not None:
            print("Stopping scheduler process...")
            self.stop_event.set()
            
            self.scheduler_process.join(timeout=10.0)
            
            if self.scheduler_process.is_alive():
                print("Scheduler process did not stop gracefully, terminating...")
                self.scheduler_process.terminate()
                self.scheduler_process.join(timeout=5.0)
                
                if self.scheduler_process.is_alive():
                    print("Force killing scheduler process...")
                    self.scheduler_process.kill()
                    self.scheduler_process.join(timeout=2.0)
            
            self.scheduler_process = None
            print("Scheduler process stopped")

    async def add_request(self, prompt: str, max_new_tokens: int):
        with self.request_id_lock:
            request_id = self.request_id_counter
            self.request_id_counter += 1
        
        request_msg = RequestMessage(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )
        self.request_queue.put(request_msg)
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        return request_id
    
    async def _response_collector(self):
        while True:
            if not self.response_queue.empty():
                response = self.response_queue.get()
                
                if response.request_id in self.pending_requests:
                    self.pending_requests[response.request_id].set_result(response)
                else:
                    print(f"Response for request_id {response.request_id} not found")
                
            await asyncio.sleep(0.01)

    async def get_result(self, request_id: int):
        future = self.pending_requests[request_id]
        return await future

    def run(self):
        # Start scheduler in background
        self.start_scheduler()

        @app.on_event("startup")
        async def startup_event():
            asyncio.create_task(self._response_collector())
        
        @app.on_event("shutdown")
        async def shutdown_event():
            self.stop_scheduler()

        @app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            request_id = await self.add_request(request.prompt, request.max_new_tokens)
            return await self.get_result(request_id)

        uvicorn.run(app, host="0.0.0.0", port=42316)    
        