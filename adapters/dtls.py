# adapters/dtls.py
import asyncio
import ssl
import struct
import time
import random
import os
from typing import List, Tuple

from .base import ProtocolAdapter
from OpenSSL import SSL

# aiocoap.transports.dtls may not exist in all versions, guard the import
try:
    from aiocoap.transports.dtls import DtlsClientTransport, MessageInterface
    AIOCOAP_DTLS_AVAILABLE = True
except ImportError:
    AIOCOAP_DTLS_AVAILABLE = False

FIXED_PACKET_SIZE = 64
BENCH_HDR_FORMAT = "!IHQQI"
BENCH_HDR_SIZE = struct.calcsize(BENCH_HDR_FORMAT)
BENCH_HDR_MAGIC = 0xDEADBEEF
BENCH_HDR_VERSION = 1

TLS13_CIPHERS = {
    "aesgcm": b"TLS_AES_128_GCM_SHA256",
    "aes256gcm": b"TLS_AES_256_GCM_SHA384",
    "chacha20": b"TLS_CHACHA20_POLY1305_SHA256",
    "aesccm": b"TLS_AES_128_CCM_SHA256",
    "aesccm8": b"TLS_AES_128_CCM_8_SHA256",
}

async def client_sender(session: "MessageInterface", size: int, rate: int, duration: int, warmup: int) -> List[Tuple[float, float]]:
    rtt_results: List[Tuple[float, float]] = []
    period = 1.0 / rate
    start_time = time.monotonic()
    seq = 0
    payload_data = random.randbytes(size - BENCH_HDR_SIZE)

    while (elapsed := time.monotonic() - start_time) < duration:
        t_send_ns = time.monotonic_ns()
        header = struct.pack(BENCH_HDR_FORMAT, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, seq, t_send_ns, len(payload_data))
        packet = header + payload_data

        try:
            session.send(packet)
            resp_payload = await asyncio.wait_for(session.read(), timeout=5.0)
            
            if elapsed > warmup and resp_payload:
                if len(resp_payload) >= BENCH_HDR_SIZE:
                    resp_magic, _, resp_seq, _, _ = struct.unpack(BENCH_HDR_FORMAT, resp_payload[:BENCH_HDR_SIZE])
                    if resp_magic == BENCH_HDR_MAGIC and resp_seq == seq:
                        rtt_ns = time.monotonic_ns() - t_send_ns
                        rtt_results.append((rtt_ns / 1e9, time.monotonic()))  # RTT in seconds
        except Exception as e:
            print(f"DTLS request failed: {e}")

        seq += 1
        await asyncio.sleep(period)
    
    return rtt_results

class DTLSAdapter(ProtocolAdapter):
    name = "dtls"

    def __init__(self):
        if not AIOCOAP_DTLS_AVAILABLE:
            raise RuntimeError("DTLS transport not available in this aiocoap version. Try installing with 'pip install aiocoap[dtls]' or a different version.")
        try:
            from OpenSSL import SSL
        except ImportError:
            raise RuntimeError("pyopenssl is not installed. Run 'pip install pyopenssl'")

    async def _run_async(self, host, port, cipher, size, rate, duration, warmup, ca):
        ssl_context = SSL.Context(SSL.DTLS_METHOD)
        cipher_bytes = TLS13_CIPHERS.get(cipher)
        if cipher_bytes:
            try: ssl_context.set_cipher_list(cipher_bytes)
            except SSL.Error as e: print(f"Warning: Failed to set DTLS ciphersuite: {e}")
        
        if ca and os.path.exists(ca):
            ssl_context.load_verify_locations(cafile=ca)
            ssl_context.set_verify(SSL.VERIFY_PEER, lambda conn, cert, errno, depth, ok: ok)
        else:
            ssl_context.set_verify(SSL.VERIFY_NONE, lambda conn, cert, errno, depth, ok: True)

        transport, session = None, None
        try:
            transport = await DtlsClientTransport.create_client_transport_endpoint(ssl_context=ssl_context)
            session = await transport.get_message_interface((host, port))
            return await client_sender(session, size, rate, duration, warmup)
        finally:
            if session: session.shutdown()
            if transport: await transport.shutdown()

    def run_load(self, host, port, cipher, size, rate, duration, warmup, ca=None, **kwargs):
        results_list = []
        def thread_target():
            rtt_results = asyncio.run(self._run_async(host, port, cipher, FIXED_PACKET_SIZE, rate, duration, warmup, ca))
            results_list.extend(rtt_results)

        thread = threading.Thread(target=thread_target, daemon=True)
        thread.start()
        return thread, results_list
