# adapters/coap_oscore.py
import asyncio
import struct
import time
import random
import os
import threading
from typing import List, Tuple

from .base import ProtocolAdapter
from .common import BENCH_HDR_FORMAT, BENCH_HDR_SIZE, BENCH_HDR_MAGIC, BENCH_HDR_VERSION
import aiocoap
import aiocoap.oscore

async def client_sender(context: aiocoap.Context, uri: str, size: int, rate: int, duration: int, warmup: int, error_counter=None, lock=None):
    """
    Send CoAP packets and collect RTT measurements.

    Args:
        context: aiocoap Context
        uri: CoAP URI to send requests to
        size: Packet size in bytes
        rate: Packets per second
        duration: Total duration in seconds
        warmup: Warmup period in seconds (skip RTT collection)
        error_counter: Dictionary for error tracking
        lock: Threading lock for error_counter
    """
    period = 1.0 / rate
    start_time = time.monotonic()
    seq = 0
    payload_data = random.randbytes(size - BENCH_HDR_SIZE)

    while (elapsed := time.monotonic() - start_time) < duration:
        t_send_ns = time.monotonic_ns()
        header = struct.pack(BENCH_HDR_FORMAT, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, seq, t_send_ns, len(payload_data))
        packet = header + payload_data
        request = aiocoap.Message(code=aiocoap.POST, uri=uri, payload=packet)

        try:
            response = await asyncio.wait_for(context.request(request).response, timeout=5.0)
            if elapsed > warmup:
                resp_payload = response.payload
                if len(resp_payload) >= BENCH_HDR_SIZE:
                    try:
                        resp_magic, _, resp_seq, _, _ = struct.unpack(BENCH_HDR_FORMAT, resp_payload[:BENCH_HDR_SIZE])
                        if resp_magic == BENCH_HDR_MAGIC and resp_seq == seq:
                            rtt_ns = time.monotonic_ns() - t_send_ns
                            if error_counter is not None and lock is not None:
                                with lock:
                                    error_counter['rtt_results'].append((rtt_ns / 1e9, time.time()))
                                    error_counter['success_count'] += 1
                        else:
                            if error_counter is not None and lock is not None:
                                with lock:
                                    error_counter['mismatch_count'] += 1
                    except struct.error:
                        if error_counter is not None and lock is not None:
                            with lock:
                                error_counter['unpack_error_count'] += 1
                else:
                    if error_counter is not None and lock is not None:
                        with lock:
                            error_counter['short_response_count'] += 1
        except asyncio.TimeoutError:
            if error_counter is not None and lock is not None:
                with lock:
                    error_counter['timeout_count'] += 1
        except Exception as e:
            if error_counter is not None and lock is not None:
                with lock:
                    error_counter['request_error_count'] += 1

        seq += 1
        await asyncio.sleep(period)

class CoAPAdapter(ProtocolAdapter):
    name = "coap"

    def __init__(self):
        try:
            import aiocoap
            import aiocoap.oscore
        except ImportError:
            raise RuntimeError("aiocoap or its oscore context is not installed. Run 'pip install aiocoap[oscore]'")

        # Thread safety
        self.lock = threading.Lock()
        self.error_counter = {
            'rtt_results': [],
            'success_count': 0,
            'mismatch_count': 0,
            'short_response_count': 0,
            'unpack_error_count': 0,
            'timeout_count': 0,
            'request_error_count': 0,
        }
        self.stop_event = threading.Event()

    async def _run_async(self, host, port, cipher, size, rate, duration, warmup, oscore_context_cfg):
        """Run async CoAP client."""
        uri = f"coap://{host}:{port}/echo"
        context = None
        try:
            if cipher == 'oscore':
                if not oscore_context_cfg:
                    raise ValueError("OSCORE requires 'oscore_context' configuration")
                key = bytes.fromhex(oscore_context_cfg['key'].replace('0x', ''))
                salt = bytes.fromhex(oscore_context_cfg['salt'].replace('0x', ''))
                sender_id = bytes.fromhex(oscore_context_cfg['sender_id'].replace('0x', ''))
                recipient_id = bytes.fromhex(oscore_context_cfg['recipient_id'].replace('0x', ''))
                secctx = aiocoap.oscore.CanBeAEAD.from_parameters(
                    (key, salt, sender_id, recipient_id)
                )
                context = await aiocoap.Context.create_client_context()
                context.client_credentials['coap://*'] = secctx
            elif cipher == 'plain':
                context = await aiocoap.Context.create_client_context()
            else:
                raise ValueError(f"Unsupported cipher for CoAP: {cipher}")

            await client_sender(context, uri, size, rate, duration, warmup, self.error_counter, self.lock)
        except Exception as e:
            with self.lock:
                self.error_counter['request_error_count'] += 1
        finally:
            if context:
                await context.shutdown()

    def run_load(self, host, port, cipher, size, rate, duration, warmup, oscore_context=None, **kwargs):
        """
        Run load test using CoAP protocol.

        Returns:
            tuple: (thread, rtt_results_list) where rtt_results_list is thread-safe
        """
        # Reset counters
        with self.lock:
            self.error_counter['rtt_results'].clear()
            self.error_counter['success_count'] = 0
            self.error_counter['mismatch_count'] = 0
            self.error_counter['short_response_count'] = 0
            self.error_counter['unpack_error_count'] = 0
            self.error_counter['timeout_count'] = 0
            self.error_counter['request_error_count'] = 0
        self.stop_event.clear()

        def thread_target():
            try:
                asyncio.run(self._run_async(host, port, cipher, size, rate, duration, warmup, oscore_context))
            except Exception as e:
                print(f"CoAP adapter thread error: {e}")
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
                'mismatch_count': self.error_counter['mismatch_count'],
                'short_response_count': self.error_counter['short_response_count'],
                'unpack_error_count': self.error_counter['unpack_error_count'],
                'timeout_count': self.error_counter['timeout_count'],
                'request_error_count': self.error_counter['request_error_count'],
            }
