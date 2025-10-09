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
    "aes256gcm": "TLS_AES_256_GCM_SHA384",
}

class MQTTAdapter(ProtocolAdapter):
    name = "mqtt"

    def __init__(self):
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise RuntimeError("paho-mqtt is not installed. Run 'pip install paho-mqtt'")
        self.rtt_results: List[float] = []
        self.sent_packets: Dict[int, int] = {}
        self.client = None
        self.stop_event = threading.Event()

    def on_message(self, client, userdata, msg):
        # --- DEBUG: Message received ---
        # print(f"MQTT message received on topic {msg.topic}")
        payload = msg.payload
        if len(payload) >= BENCH_HDR_SIZE:
            try:
                magic, _, seq, resp_t_send_ns, _ = struct.unpack(BENCH_HDR_FORMAT, payload[:BENCH_HDR_SIZE])
                # --- DEBUG: Unpacked data ---
                # print(f"  -> Unpacked: magic={hex(magic)}, seq={seq}")
                if magic == BENCH_HDR_MAGIC and seq in self.sent_packets:
                    # --- DEBUG: Packet match found ---
                    # print(f"  -> Match found for seq={seq}!")
                    t_send_ns = self.sent_packets.pop(seq)
                    rtt_ns = time.monotonic_ns() - t_send_ns
                    self.rtt_results.append(rtt_ns / 1e9)
                # else:
                    # --- DEBUG: Packet mismatch ---
                    # print(f"  -> No match for seq={seq}. Sent keys: {list(self.sent_packets.keys())[:5]}...")
            except (struct.error, KeyError):
                # print("  -> Error unpacking or processing packet.")
                pass

    def _publisher_thread(self, size: int, rate: int, duration: int, warmup: int):
        start_time = time.monotonic()
        period = 1.0 / rate
        seq = 0
        payload_data = random.randbytes(size - BENCH_HDR_SIZE)

        while not self.stop_event.is_set() and time.monotonic() - start_time < duration:
            t_send_ns = time.monotonic_ns()
            header = struct.pack(BENCH_HDR_FORMAT, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, seq, t_send_ns, len(payload_data))
            packet = header + payload_data
            
            if time.monotonic() - start_time > warmup:
                self.sent_packets[seq] = t_send_ns

            self.client.publish("bench/echo", packet, qos=0)
            seq += 1
            time.sleep(max(0, period - ((time.monotonic_ns() - t_send_ns) / 1e9)))

    def run_load(self, host, port, cipher, size, rate, duration, warmup, ca=None, cert=None, key=None, **kwargs):
        self.rtt_results.clear()
        self.sent_packets.clear()
        self.stop_event.clear()

        self.client = mqtt.Client()
        self.client.on_message = self.on_message

        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        if ca and os.path.exists(ca):
            context.load_verify_locations(cafile=ca)
        
        cipher_str = TLS13_CIPHERS.get(cipher)
        if cipher_str and hasattr(context, "set_ciphersuites"):
            try: context.set_ciphersuites(cipher_str)
            except ssl.SSLError as e: print(f"Warning: {e}")

        if cert and key: context.load_cert_chain(certfile=cert, keyfile=key)
        self.client.tls_set_context(context)

        try:
            self.client.connect(host, port, 60)
        except Exception as e:
            raise RuntimeError(f"MQTT connection failed: {e}")

        self.client.subscribe("bench/echo/response", qos=0)
        self.client.loop_start()

        pub_thread = threading.Thread(target=self._publisher_thread, args=(size, rate, duration, warmup), daemon=True)
        pub_thread.start()

        return pub_thread, self.rtt_results

    def stop_load(self):
        self.stop_event.set()
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
