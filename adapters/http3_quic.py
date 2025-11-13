# adapters/http3_quic.py
import asyncio
import struct
import time
import random
import os
import threading
from typing import Deque, Dict, Optional, List, cast, Tuple
from collections import deque

from .base import ProtocolAdapter
from .common import BENCH_HDR_FORMAT, BENCH_HDR_SIZE, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, TLS13_CIPHERS
from aioquic.asyncio.client import connect
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import DataReceived, H3Event
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent

class H3ClientProtocol(QuicConnectionProtocol):
    def __init__(self, *args, error_counter=None, lock=None, warmup=0, start_time=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._http: Optional[H3Connection] = None
        self._sent_packets: Dict[int, int] = {}
        self._error_counter = error_counter
        self._lock = lock
        self._warmup = warmup
        self._start_time = start_time

    def quic_event_received(self, event: QuicEvent):
        if self._http is None:
            self._http = H3Connection(self._quic)

        for h3_event in self._http.handle_event(event):
            if isinstance(h3_event, DataReceived):
                if h3_event.stream_id in self._sent_packets:
                    t_send_ns = self._sent_packets[h3_event.stream_id]
                    elapsed = time.monotonic() - self._start_time

                    if len(h3_event.data) >= BENCH_HDR_SIZE:
                        try:
                            magic, _, seq, _, _ = struct.unpack(BENCH_HDR_FORMAT, h3_event.data[:BENCH_HDR_SIZE])
                            if magic == BENCH_HDR_MAGIC:
                                rtt_ns = time.monotonic_ns() - t_send_ns
                                if elapsed > self._warmup and self._error_counter is not None and self._lock is not None:
                                    with self._lock:
                                        self._error_counter['rtt_results'].append((rtt_ns / 1e9, time.time()))
                                        self._error_counter['success_count'] += 1
                                del self._sent_packets[h3_event.stream_id]
                            else:
                                if self._error_counter is not None and self._lock is not None:
                                    with self._lock:
                                        self._error_counter['mismatch_count'] += 1
                        except struct.error:
                            if self._error_counter is not None and self._lock is not None:
                                with self._lock:
                                    self._error_counter['unpack_error_count'] += 1
                    else:
                        if self._error_counter is not None and self._lock is not None:
                            with self._lock:
                                self._error_counter['short_response_count'] += 1

    def record_sent_packet(self, stream_id: int, t_send_ns: int):
        self._sent_packets[stream_id] = t_send_ns

async def client_sender(protocol: H3ClientProtocol, host: str, size: int, rate: int, duration: int):
    """
    Send HTTP/3 packets and collect RTT measurements.
    """
    http = protocol._http
    period = 1.0 / rate
    protocol._start_time = time.monotonic()
    seq = 0
    payload_data = random.randbytes(size - BENCH_HDR_SIZE)

    while (elapsed := time.monotonic() - protocol._start_time) < duration:
        stream_id = protocol._quic.get_next_available_stream_id()
        t_send_ns = time.monotonic_ns()
        header = struct.pack(BENCH_HDR_FORMAT, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, seq, t_send_ns, len(payload_data))
        packet = header + payload_data

        http.send_headers(stream_id=stream_id, headers=[
            (b":method", b"POST"), (b":scheme", b"https"),
            (b":authority", host.encode()), (b":path", b"/echo"),
            (b"content-length", str(len(packet)).encode()),
        ])
        http.send_data(stream_id=stream_id, data=packet, end_stream=True)

        protocol.record_sent_packet(stream_id, t_send_ns)

        protocol.transmit()
        seq += 1
        await asyncio.sleep(period)

    await asyncio.sleep(2)  # Wait for final responses
    protocol.close()

class HTTP3Adapter(ProtocolAdapter):
    name = "http3"

    def __init__(self):
        try:
            import aioquic
        except ImportError:
            raise RuntimeError("aioquic is not installed. Run 'pip install aioquic'")

        # Thread safety
        self.lock = threading.Lock()
        self.error_counter = {
            'rtt_results': [],
            'success_count': 0,
            'mismatch_count': 0,
            'short_response_count': 0,
            'unpack_error_count': 0,
        }
        self.stop_event = threading.Event()

    async def _run_async(self, host, port, cipher, size, rate, duration, warmup, ca):
        """Run async HTTP/3 client."""
        config = QuicConfiguration(is_client=True, alpn_protocols=H3_ALPN)
        if ca and os.path.exists(ca):
            config.load_verify_locations(cafile=ca)

        try:
            def create_protocol_with_params(*args, **kwargs):
                return H3ClientProtocol(
                    *args,
                    error_counter=self.error_counter,
                    lock=self.lock,
                    warmup=warmup,
                    start_time=None,
                    **kwargs
                )

            async with connect(host, port, configuration=config, create_protocol=create_protocol_with_params) as protocol:
                protocol = cast(H3ClientProtocol, protocol)
                await client_sender(protocol, host, size, rate, duration)
        except (ConnectionError, TimeoutError, OSError) as e:
            with self.lock:
                self.error_counter['other_error_count'] += 1
        except Exception as e:
            with self.lock:
                self.error_counter['other_error_count'] += 1

    def run_load(self, host, port, cipher, size, rate, duration, warmup, ca=None, **kwargs):
        """
        Run load test using HTTP/3 protocol.

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
        self.stop_event.clear()

        def thread_target():
            try:
                asyncio.run(self._run_async(host, port, cipher, size, rate, duration, warmup, ca))
            except Exception as e:
                print(f"HTTP/3 adapter thread error: {e}")
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
            }
