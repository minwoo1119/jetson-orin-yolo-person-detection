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
from aioquic.asyncio.client import connect
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import DataReceived, H3Event
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent

FIXED_PACKET_SIZE = 64
BENCH_HDR_FORMAT = "!IHQQI"
BENCH_HDR_SIZE = struct.calcsize(BENCH_HDR_FORMAT)
BENCH_HDR_MAGIC = 0xDEADBEEF
BENCH_HDR_VERSION = 1

class H3ClientProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._http: Optional[H3Connection] = None
        self.rtt_results: Deque[Tuple[float, float]] = deque()
        self._sent_packets: Dict[int, int] = {}
        self._start_time: float = 0

    def quic_event_received(self, event: QuicEvent):
        if self._http is None:
            self._http = H3Connection(self._quic)
        
        for h3_event in self._http.handle_event(event):
            if isinstance(h3_event, DataReceived):
                if h3_event.stream_id in self._sent_packets:
                    t_send_ns = self._sent_packets[h3_event.stream_id]
                    if len(h3_event.data) >= BENCH_HDR_SIZE:
                        try:
                            magic, _, seq, _, _ = struct.unpack(BENCH_HDR_FORMAT, h3_event.data[:BENCH_HDR_SIZE])
                            if magic == BENCH_HDR_MAGIC:
                                rtt_ns = time.monotonic_ns() - t_send_ns
                                self.rtt_results.append((rtt_ns / 1e9, time.monotonic()))
                                del self._sent_packets[h3_event.stream_id]
                        except struct.error:
                            pass

    def record_sent_packet(self, stream_id: int, t_send_ns: int):
        self._sent_packets[stream_id] = t_send_ns

async def client_sender(protocol: H3ClientProtocol, host: str, size: int, rate: int, duration: int, warmup: int):
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
        
        if elapsed > warmup:
            protocol.record_sent_packet(stream_id, t_send_ns)

        protocol.transmit()
        seq += 1
        await asyncio.sleep(period)

    await asyncio.sleep(2)
    protocol.close()

class HTTP3Adapter(ProtocolAdapter):
    name = "http3"

    def __init__(self):
        try: import aioquic
        except ImportError: raise RuntimeError("aioquic is not installed. Run 'pip install aioquic'")

    async def _run_async(self, host, port, cipher, size, rate, duration, warmup, ca):
        config = QuicConfiguration(is_client=True, alpn_protocols=H3_ALPN)
        if ca and os.path.exists(ca):
            config.load_verify_locations(cafile=ca)

        print(f"[DEBUG] HTTP3/QUIC attempting connection to {host}:{port}")
        try:
            async with connect(host, port, configuration=config, create_protocol=H3ClientProtocol) as protocol:
                protocol = cast(H3ClientProtocol, protocol)
                print(f"[SUCCESS] HTTP3/QUIC connected to {host}:{port}")
                await client_sender(protocol, host, size, rate, duration, warmup)
                return list(protocol.rtt_results)
        except ConnectionError as e:
            print(f"[ERROR] HTTP3/QUIC ConnectionError: {e}")
            return []
        except TimeoutError as e:
            print(f"[ERROR] HTTP3/QUIC timeout: {e}")
            return []
        except OSError as e:
            print(f"[ERROR] HTTP3/QUIC network error: {e}")
            return []
        except Exception as e:
            print(f"[ERROR] HTTP3/QUIC failed with {type(e).__name__}: {e}")
            return []

    def run_load(self, host, port, cipher, size, rate, duration, warmup, ca=None, **kwargs):
        results_list = []
        def thread_target():
            rtt_results = asyncio.run(self._run_async(host, port, cipher, FIXED_PACKET_SIZE, rate, duration, warmup, ca))
            results_list.extend(rtt_results)

        thread = threading.Thread(target=thread_target, daemon=True)
        thread.start()
        return thread, results_list