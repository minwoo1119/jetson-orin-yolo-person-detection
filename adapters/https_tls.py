# adapters/https_tls.py
import asyncio
import ssl
import struct
import time
import random
import os
import threading
from typing import List

from .base import ProtocolAdapter
import httpx

# Packet header definition
BENCH_HDR_FORMAT = "!IHQQI"
BENCH_HDR_SIZE = struct.calcsize(BENCH_HDR_FORMAT)
BENCH_HDR_MAGIC = 0xDEADBEEF
BENCH_HDR_VERSION = 1

# TLS 1.3 Cipher Suite names
TLS13_CIPHERS = {
    "aesgcm": "TLS_AES_128_GCM_SHA256",
    "chacha20": "TLS_CHACHA20_POLY1305_SHA256",
    "aesccm8": "TLS_AES_128_CCM_8_SHA256",
}

async def client_sender(client: httpx.AsyncClient, url: str, size: int, rate: int, duration: int, warmup: int) -> List[float]:
    rtt_results: List[float] = []
    period = 1.0 / rate
    start_time = time.monotonic()
    seq = 0
    payload_data = random.randbytes(size - BENCH_HDR_SIZE)

    while time.monotonic() - start_time < duration:
        t_send_ns = time.monotonic_ns()
        header = struct.pack(BENCH_HDR_FORMAT, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, seq, t_send_ns, len(payload_data))
        packet = header + payload_data

        try:
            response = await client.post(url, content=packet, timeout=5)
            response.raise_for_status()
            if time.monotonic() - start_time > warmup:
                resp_payload = response.content
                if len(resp_payload) >= BENCH_HDR_SIZE:
                    resp_magic, _, resp_seq, resp_t_send_ns, _ = struct.unpack(BENCH_HDR_FORMAT, resp_payload[:BENCH_HDR_SIZE])
                    if resp_magic == BENCH_HDR_MAGIC and resp_seq == seq:
                        rtt_ns = time.monotonic_ns() - resp_t_send_ns
                        rtt_results.append(rtt_ns / 1e9)
        except httpx.RequestError as e:
            print(f"HTTPS request failed: {e}")
        
        seq += 1
        await asyncio.sleep(period)
    
    return rtt_results

class HTTPSAdapter(ProtocolAdapter):
    name = "https"

    def __init__(self):
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx is not installed. Run 'pip install httpx'")

    async def _run_async(self, host, port, cipher, size, rate, duration, warmup, ca):
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
            return await client_sender(client, url, size, rate, duration, warmup)

    def run_load(self, host, port, cipher, size, rate, duration, warmup, ca=None, **kwargs):
        results_list = []
        def thread_target():
            rtt_results = asyncio.run(self._run_async(host, port, cipher, size, rate, duration, warmup, ca))
            results_list.extend(rtt_results)

        thread = threading.Thread(target=thread_target, daemon=True)
        thread.start()
        return thread, results_list
