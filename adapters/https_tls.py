# adapters/https_tls.py
import asyncio
import ssl
import time
import random
import os
import threading
from typing import List, Tuple

from .base import ProtocolAdapter
from .common import BENCH_HDR_FORMAT, BENCH_HDR_SIZE, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, TLS13_CIPHERS
import httpx

async def client_sender(client: httpx.AsyncClient, url: str, size: int, rate: int, duration: int, warmup: int, error_counter: dict, lock: threading.Lock) -> None:
    """
    Send packets to server and collect RTT measurements.

    Args:
        client: HTTP client
        url: Target URL
        size: Packet size
        rate: Send rate (packets per second)
        duration: Test duration
        warmup: Warmup period before collecting RTT
        error_counter: Dictionary to track errors (shared with thread)
        lock: Thread lock for safe access to shared data
    """
    period = 1.0 / rate
    start_time = time.monotonic()
    seq = 0
    payload_data = random.randbytes(size - BENCH_HDR_SIZE)

    while time.monotonic() - start_time < duration:
        t_send_ns = time.monotonic_ns()
        header = BENCH_HDR_FORMAT
        import struct
        packet_header = struct.pack(header, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, seq, t_send_ns, len(payload_data))
        packet = packet_header + payload_data

        try:
            response = await client.post(url, content=packet, timeout=5)
            response.raise_for_status()
            if time.monotonic() - start_time > warmup:
                resp_payload = response.content
                if len(resp_payload) >= BENCH_HDR_SIZE:
                    resp_magic, _, resp_seq, resp_t_send_ns, _ = struct.unpack(BENCH_HDR_FORMAT, resp_payload[:BENCH_HDR_SIZE])
                    if resp_magic == BENCH_HDR_MAGIC and resp_seq == seq:
                        rtt_ns = time.monotonic_ns() - t_send_ns
                        with lock:
                            error_counter['rtt_results'].append((rtt_ns / 1e9, time.time()))
                            error_counter['success_count'] += 1
                    else:
                        with lock:
                            error_counter['mismatch_count'] += 1
                else:
                    with lock:
                        error_counter['short_response_count'] += 1
        except httpx.TimeoutException as e:
            with lock:
                error_counter['timeout_count'] += 1
        except httpx.RequestError as e:
            with lock:
                error_counter['request_error_count'] += 1
        except Exception as e:
            with lock:
                error_counter['other_error_count'] += 1

        seq += 1
        await asyncio.sleep(period)

class HTTPSAdapter(ProtocolAdapter):
    name = "https"

    def __init__(self):
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx is not installed. Run 'pip install httpx'")

        self.lock = threading.Lock()
        self.error_counter = {
            'rtt_results': [],
            'success_count': 0,
            'timeout_count': 0,
            'request_error_count': 0,
            'mismatch_count': 0,
            'short_response_count': 0,
            'other_error_count': 0,
        }
        self.stop_event = threading.Event()

    async def _run_async(self, host, port, cipher, size, rate, duration, warmup, ca):
        """Run async HTTP client sender."""
        url = f"https://{host}:{port}/echo"
        context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
        if ca and os.path.exists(ca):
            context.load_verify_locations(cafile=ca)

        cipher_str = TLS13_CIPHERS.get(cipher)
        if cipher_str and hasattr(context, "set_ciphersuites"):
            try:
                context.set_ciphersuites(cipher_str)
            except ssl.SSLError as e:
                print(f"Warning: Failed to set TLS 1.3 ciphersuite for HTTPS: {e}")

        async with httpx.AsyncClient(verify=context, http2=False) as client:
            await client_sender(client, url, size, rate, duration, warmup, self.error_counter, self.lock)

    def run_load(self, host, port, cipher, size, rate, duration, warmup, ca=None, **kwargs):
        """
        Run load test using HTTPS protocol.

        Returns:
            tuple: (thread, rtt_results_list) where rtt_results_list is thread-safe
        """
        # Reset counters
        with self.lock:
            self.error_counter['rtt_results'].clear()
            self.error_counter['success_count'] = 0
            self.error_counter['timeout_count'] = 0
            self.error_counter['request_error_count'] = 0
            self.error_counter['mismatch_count'] = 0
            self.error_counter['short_response_count'] = 0
            self.error_counter['other_error_count'] = 0
        self.stop_event.clear()

        def thread_target():
            try:
                asyncio.run(self._run_async(host, port, cipher, size, rate, duration, warmup, ca))
            except Exception as e:
                print(f"HTTPS adapter thread error: {e}")
            finally:
                self.stop_event.set()

        thread = threading.Thread(target=thread_target, daemon=False)
        thread.start()

        # Return thread and reference to results list (thread-safe with lock)
        return thread, self.error_counter['rtt_results']

    def stop_load(self):
        """Stop the load test gracefully."""
        self.stop_event.set()

    def get_error_stats(self):
        """Get error statistics (thread-safe)."""
        with self.lock:
            return {
                'success_count': self.error_counter['success_count'],
                'timeout_count': self.error_counter['timeout_count'],
                'request_error_count': self.error_counter['request_error_count'],
                'mismatch_count': self.error_counter['mismatch_count'],
                'short_response_count': self.error_counter['short_response_count'],
                'other_error_count': self.error_counter['other_error_count'],
            }