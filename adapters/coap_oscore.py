# adapters/coap_oscore.py
import asyncio
import struct
import time
import random
import os
import threading
from typing import List, Tuple

from .base import ProtocolAdapter
import aiocoap
import aiocoap.oscore

FIXED_PACKET_SIZE = 64
BENCH_HDR_FORMAT = "!IHQQI"
BENCH_HDR_SIZE = struct.calcsize(BENCH_HDR_FORMAT)
BENCH_HDR_MAGIC = 0xDEADBEEF
BENCH_HDR_VERSION = 1

async def client_sender(context: aiocoap.Context, uri: str, size: int, rate: int, duration: int, warmup: int) -> List[Tuple[float, float]]:
    rtt_results: List[Tuple[float, float]] = []
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
                    resp_magic, _, resp_seq, _, _ = struct.unpack(BENCH_HDR_FORMAT, resp_payload[:BENCH_HDR_SIZE])
                    if resp_magic == BENCH_HDR_MAGIC and resp_seq == seq:
                        rtt_ns = time.monotonic_ns() - t_send_ns
                        rtt_results.append((rtt_ns / 1e9, time.monotonic()))  # RTT in seconds
        except Exception as e:
            print(f"CoAP request failed: {e}")

        seq += 1
        await asyncio.sleep(period)
    
    return rtt_results

class CoAPAdapter(ProtocolAdapter):
    name = "coap"

    def __init__(self):
        try:
            import aiocoap
            import aiocoap.oscore
        except ImportError:
            raise RuntimeError("aiocoap or its oscore context is not installed. Run 'pip install aiocoap[oscore]'")

    async def _run_async(self, host, port, cipher, size, rate, duration, warmup, oscore_context_cfg):
        uri = f"coap://{host}:{port}/echo"
        context = None
        try:
            if cipher == 'oscore':
                if not oscore_context_cfg: raise ValueError("OSCORE requires 'oscore_context' configuration")
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

            return await client_sender(context, uri, size, rate, duration, warmup)
        finally:
            if context: await context.shutdown()

    def run_load(self, host, port, cipher, size, rate, duration, warmup, oscore_context=None, **kwargs):
        results_list = []
        def thread_target():
            rtt_results = asyncio.run(self._run_async(host, port, cipher, size, rate, duration, warmup, oscore_context))
            results_list.extend(rtt_results)

        thread = threading.Thread(target=thread_target, daemon=True)
        thread.start()
        return thread, results_list
