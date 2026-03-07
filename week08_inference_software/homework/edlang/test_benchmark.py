#!/usr/bin/env python3

import asyncio
import random
import statistics
import string
import threading
import time
from typing import Any, Dict, Optional

import aiohttp


def generate_random_prompt(length: int) -> str:
    """Generate random prompt of given character length."""
    chars = string.ascii_letters + string.digits + " ,.!?;:"
    words = [
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
        "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
        "man", "new", "now", "old", "see", "two", "way", "who", "boy", "did",
        "its", "let", "put", "say", "she", "too", "use", "data", "model", "system",
        "process", "function", "algorithm", "network", "learning", "training",
        "inference", "server", "request", "response", "token", "prompt", "generate"
    ]

    prompt_parts = []
    current_length = 0
    while current_length < length * 0.7:
        word = random.choice(words)
        if current_length + len(word) + 1 <= length:
            prompt_parts.append(word)
            current_length += len(word) + 1
        else:
            break

    remaining = length - current_length
    if remaining > 0:
        prompt_parts.append("".join(random.choice(chars) for _ in range(remaining)))

    prompt = " ".join(prompt_parts)
    if len(prompt) > length:
        prompt = prompt[:length]
    elif len(prompt) < length:
        prompt += "".join(random.choice(chars) for _ in range(length - len(prompt)))
    return prompt


def get_mode_config(mode: str, prompt_length: Optional[int] = None) -> Dict[str, Any]:
    """Get prompt length and max_new_tokens for a given mode."""
    configs = {
        "hard_prefill": {"max_new_tokens": 20, "prompt_length": prompt_length or 2000},
        "hard_decode": {"max_new_tokens": 2000, "prompt_length": prompt_length or 50},
        "medium": {"max_new_tokens": 200, "prompt_length": prompt_length or 200},
    }
    return configs.get(mode, configs["medium"])


async def _send_request(
    session: aiohttp.ClientSession,
    request_id: int,
    prompt: str,
    max_new_tokens: int,
    url: str,
) -> Dict[str, Any]:
    """Send single async request and return result."""
    start = time.perf_counter()
    try:
        async with session.post(
            url,
            json={"prompt": prompt, "max_new_tokens": max_new_tokens},
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            latency = time.perf_counter() - start
            if resp.status == 200:
                data = await resp.json()
                tokens = data.get("generated_tokens", [])
                return {
                    "request_id": request_id,
                    "response": data.get("generated_text", ""),
                    "latency": latency,
                    "num_tokens": len(tokens),
                }
            return {
                "request_id": request_id,
                "response": None,
                "latency": latency,
                "num_tokens": 0,
                "error": f"Status {resp.status}",
            }
    except Exception as e:
        latency = time.perf_counter() - start
        return {
            "request_id": request_id,
            "response": None,
            "latency": latency,
            "num_tokens": 0,
            "error": str(e),
        }


async def _run_benchmark(
    mode: str,
    rps: float,
    num_requests: int,
    url: str,
    warmup: int,
    prompt_length: Optional[int],
) -> Dict[str, Any]:
    """Async benchmark implementation."""
    config = get_mode_config(mode, prompt_length)
    max_new_tokens = config["max_new_tokens"]
    prompt_len = config["prompt_length"]
    delay = 1.0 / rps if rps > 0 else 0

    print(f"[Benchmark] mode={mode}, target_rps={rps}, num_requests={num_requests}")
    print(f"[Benchmark] prompt_len={prompt_len} chars, max_new_tokens={max_new_tokens}")

    async with aiohttp.ClientSession() as session:
        # Warmup
        if warmup > 0:
            print(f"[Benchmark] Warmup: {warmup} requests...")
            warmup_prompt = generate_random_prompt(prompt_len)
            for i in range(warmup):
                await _send_request(session, -1, warmup_prompt, max_new_tokens, url)
            print(f"[Benchmark] Warmup done.")

        # Send requests at target RPS (no artificial concurrency limit)
        print(f"[Benchmark] Sending {num_requests} requests at {rps} RPS (delay={delay:.3f}s between starts)...")
        send_start = time.perf_counter()
        tasks = []

        for i in range(num_requests):
            await asyncio.sleep(delay)
            prompt = generate_random_prompt(prompt_len)
            task = asyncio.create_task(
                _send_request(session, i, prompt, max_new_tokens, url)
            )
            tasks.append(task)

            # Progress log every 10% or every 20 requests
            if (i + 1) % max(1, num_requests // 10) == 0 or i + 1 == num_requests:
                elapsed = time.perf_counter() - send_start
                actual_rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  -> Sent {i + 1}/{num_requests} (rate: {actual_rate:.1f} req/s)")

        send_time = time.perf_counter() - send_start
        print(f"[Benchmark] All requests sent in {send_time:.2f}s. Waiting for responses...")

        # Wait for all responses
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - send_start

    # Sort by request_id
    results = sorted(results, key=lambda x: x["request_id"])
    successful = [r for r in results if r.get("response") is not None]
    latencies = [r["latency"] for r in results]
    num_tokens_list = [r["num_tokens"] for r in results if r.get("num_tokens", 0) > 0]

    # Percentiles (safe for small lists)
    def pct(sorted_arr: list, p: float) -> float:
        if not sorted_arr:
            return 0.0
        idx = min(int(len(sorted_arr) * p / 100), len(sorted_arr) - 1)
        return sorted_arr[idx]

    sorted_lat = sorted(latencies)

    stats = {
        "total_requests": num_requests,
        "successful_requests": len(successful),
        "failed_requests": num_requests - len(successful),
        "total_time": total_time,
        "target_rps": rps,
        "submission_rps": num_requests / send_time if send_time > 0 else 0,
        "actual_rps": len(successful) / total_time if total_time > 0 else 0,
        "latency": {
            "avg": statistics.mean(latencies) if latencies else 0,
            "median": statistics.median(latencies) if latencies else 0,
            "min": min(latencies) if latencies else 0,
            "max": max(latencies) if latencies else 0,
            "p95": pct(sorted_lat, 95),
            "p99": pct(sorted_lat, 99),
        },
        "tokens": {
            "avg": statistics.mean(num_tokens_list) if num_tokens_list else 0,
            "median": statistics.median(num_tokens_list) if num_tokens_list else 0,
            "min": min(num_tokens_list) if num_tokens_list else 0,
            "max": max(num_tokens_list) if num_tokens_list else 0,
            "total": sum(num_tokens_list),
        },
    }

    print(f"[Benchmark] Done. Success: {len(successful)}/{num_requests}")
    print(f"  latency: avg={stats['latency']['avg']*1000:.0f}ms, p95={stats['latency']['p95']*1000:.0f}ms")
    print(f"  throughput: {stats['actual_rps']:.2f} req/s")

    return {
        "mode": mode,
        "requests": results,
        "statistics": stats,
    }


def benchmark(
    mode: str,
    rps: float,
    num_requests: int = 10,
    url: str = "http://localhost:42316/generate",
    warmup: int = 2,
    max_workers: Optional[int] = None,  # Ignored, kept for API compatibility
    prompt_length: Optional[int] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    def run_in_thread() -> None:
        nonlocal result
        result = asyncio.run(
            _run_benchmark(mode, rps, num_requests, url, warmup, prompt_length)
        )

    try:
        asyncio.get_running_loop()
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        return result
    except RuntimeError:
        return asyncio.run(
            _run_benchmark(mode, rps, num_requests, url, warmup, prompt_length)
        )
