# adapters/mqtt_tls.py
import ssl
import time
import threading
import struct
import random
import os
from typing import List, Dict

from .base import ProtocolAdapter
import paho.mqtt.client as mqtt

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

def on_message_rtt(client, userdata: Dict, msg):
    payload = msg.payload
    if len(payload) >= BENCH_HDR_SIZE:
        try:
            magic, _, seq, resp_t_send_ns, _ = struct.unpack(BENCH_HDR_FORMAT, payload[:BENCH_HDR_SIZE])
            if magic == BENCH_HDR_MAGIC and seq in userdata['sent_packets']:
                t_send_ns = userdata['sent_packets'].pop(seq)
                rtt_ns = time.monotonic_ns() - t_send_ns
                userdata['rtt_results'].append(rtt_ns / 1e9)
        except (struct.error, KeyError):
            pass

class MQTTAdapter(ProtocolAdapter):
    name = "mqtt"

    def __init__(self):
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise RuntimeError("paho-mqtt is not installed. Run 'pip install paho-mqtt'")

    def _make_ssl_context(self, cipher: str, ca: str, cert: str, key: str) -> ssl.SSLContext:
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        if ca and os.path.exists(ca):
            context.load_verify_locations(cafile=ca)
        cipher_str = TLS13_CIPHERS.get(cipher)
        if cipher_str and hasattr(context, "set_ciphersuites"):
            try: context.set_ciphersuites(cipher_str)
            except ssl.SSLError as e: print(f"Warning: Failed to set TLS 1.3 ciphersuite: {e}")
        if cert and key: context.load_cert_chain(certfile=cert, keyfile=key)
        return context

    def run_load(self, host, port, cipher, size, rate, duration, warmup, ca=None, cert=None, key=None, **kwargs):
        shared_data = {
            'rtt_results': [],
            'sent_packets': {}
        }

        client = mqtt.Client(userdata=shared_data)
        client.on_message = on_message_rtt
        ssl_context = self._make_ssl_context(cipher, ca, cert, key)
        client.tls_set_context(ssl_context)

        try:
            client.connect(host, port, 60)
        except Exception as e:
            raise RuntimeError(f"MQTT connection failed: {e}")

        client.subscribe("bench/echo/response", qos=0)
        
        # --- Main Loop Control ---
        start_time = time.monotonic()
        period = 1.0 / rate
        seq = 0
        payload_data = random.randbytes(size - BENCH_HDR_SIZE)
        stop_event = threading.Event()

        def test_loop():
            nonlocal seq
            while not stop_event.is_set() and time.monotonic() - start_time < duration:
                elapsed = time.monotonic() - start_time
                t_send_ns = time.monotonic_ns()
                header = struct.pack(BENCH_HDR_FORMAT, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, seq, t_send_ns, len(payload_data))
                packet = header + payload_data

                if elapsed > warmup:
                    shared_data['sent_packets'][seq] = t_send_ns
                
                client.publish("bench/echo", packet, qos=0)
                
                # Give time for the network loop to process messages
                client.loop(timeout=0.01)

                seq += 1
                time.sleep(period)
            
            # Final loop to catch remaining messages
            final_collection_start = time.time()
            while time.time() - final_collection_start < 2:
                client.loop(timeout=0.1)

            client.disconnect()
            client.loop_stop()

        thread = threading.Thread(target=test_loop, daemon=True)
        thread.start()

        return thread, shared_data['rtt_results']