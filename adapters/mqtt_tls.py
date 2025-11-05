# adapters/mqtt_tls.py
import ssl
import time
import threading
import struct
import random
import os
from typing import List, Dict

from .base import ProtocolAdapter
from .common import BENCH_HDR_FORMAT, BENCH_HDR_SIZE, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, TLS13_CIPHERS
import paho.mqtt.client as mqtt

class MQTTAdapter(ProtocolAdapter):
    name = "mqtt"

    def __init__(self):
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise RuntimeError("paho-mqtt is not installed. Run 'pip install paho-mqtt'")

        self.rtt_results: List[tuple] = []
        self.sent_packets: Dict[int, int] = {}
        self.client = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        # Error tracking
        self.error_counter = {
            'success_count': 0,
            'mismatch_count': 0,
            'short_response_count': 0,
            'unpack_error_count': 0,
        }

    def on_message(self, client, userdata, msg):
        """Handle incoming MQTT message (response from server)."""
        payload = msg.payload
        if len(payload) >= BENCH_HDR_SIZE:
            try:
                magic, _, seq, resp_t_send_ns, _ = struct.unpack(BENCH_HDR_FORMAT, payload[:BENCH_HDR_SIZE])

                with self.lock:
                    if magic == BENCH_HDR_MAGIC and seq in self.sent_packets:
                        t_send_ns = self.sent_packets.pop(seq)
                        rtt_ns = time.monotonic_ns() - t_send_ns
                        self.rtt_results.append((rtt_ns / 1e9, time.time()))
                        self.error_counter['success_count'] += 1
                    else:
                        self.error_counter['mismatch_count'] += 1
            except (struct.error, KeyError) as e:
                with self.lock:
                    self.error_counter['unpack_error_count'] += 1
        else:
            with self.lock:
                self.error_counter['short_response_count'] += 1

    def _publisher_thread(self, size: int, rate: int, duration: int, warmup: int):
        """Publisher thread that sends packets at specified rate."""
        start_time = time.monotonic()
        period = 1.0 / rate
        seq = 0
        payload_data = random.randbytes(size - BENCH_HDR_SIZE)

        while not self.stop_event.is_set() and time.monotonic() - start_time < duration:
            t_send_ns = time.monotonic_ns()
            header = struct.pack(BENCH_HDR_FORMAT, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, seq, t_send_ns, len(payload_data))
            packet = header + payload_data

            if time.monotonic() - start_time > warmup:
                with self.lock:
                    self.sent_packets[seq] = t_send_ns

            self.client.publish("bench/echo", packet, qos=0)
            seq += 1
            time.sleep(max(0, period - ((time.monotonic_ns() - t_send_ns) / 1e9)))

    def run_load(self, host, port, cipher, size, rate, duration, warmup, ca=None, cert=None, key=None, **kwargs):
        """
        Run load test using MQTT protocol.

        Returns:
            tuple: (thread, rtt_results_list) where rtt_results_list is thread-safe
        """
        # Reset state with lock
        with self.lock:
            self.rtt_results.clear()
            self.sent_packets.clear()
            self.error_counter['success_count'] = 0
            self.error_counter['mismatch_count'] = 0
            self.error_counter['short_response_count'] = 0
            self.error_counter['unpack_error_count'] = 0
        self.stop_event.clear()

        self.client = mqtt.Client()
        self.client.on_message = self.on_message

        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        if ca and os.path.exists(ca):
            context.load_verify_locations(cafile=ca)

        cipher_str = TLS13_CIPHERS.get(cipher)
        if cipher_str and hasattr(context, "set_ciphersuites"):
            try:
                context.set_ciphersuites(cipher_str)
            except ssl.SSLError as e:
                print(f"Warning: Failed to set TLS 1.3 ciphersuite for MQTT: {e}")

        if cert and key:
            context.load_cert_chain(certfile=cert, keyfile=key)
        self.client.tls_set_context(context)

        print(f"[DEBUG] MQTT attempting connection to {host}:{port}")
        try:
            self.client.connect(host, port, 60)
            print(f"[SUCCESS] MQTT connected to {host}:{port}")
        except ConnectionRefusedError as e:
            print(f"[ERROR] MQTT connection refused: {e}")
            raise RuntimeError(f"MQTT connection failed: {e}")
        except TimeoutError as e:
            print(f"[ERROR] MQTT connection timeout: {e}")
            raise RuntimeError(f"MQTT connection failed: timed out")
        except OSError as e:
            print(f"[ERROR] MQTT network error: {e}")
            raise RuntimeError(f"MQTT connection failed: {e}")
        except Exception as e:
            print(f"[ERROR] MQTT connection failed with {type(e).__name__}: {e}")
            raise RuntimeError(f"MQTT connection failed: {e}")

        self.client.subscribe("bench/echo/response", qos=0)
        self.client.loop_start()

        pub_thread = threading.Thread(target=self._publisher_thread, args=(size, rate, duration, warmup), daemon=False)
        pub_thread.start()

        return pub_thread, self.rtt_results

    def stop_load(self):
        """Stop the load test gracefully."""
        self.stop_event.set()
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()

    def get_error_stats(self):
        """Get error statistics (thread-safe)."""
        with self.lock:
            return {
                'success_count': self.error_counter['success_count'],
                'mismatch_count': self.error_counter['mismatch_count'],
                'short_response_count': self.error_counter['short_response_count'],
                'unpack_error_count': self.error_counter['unpack_error_count'],
            }
