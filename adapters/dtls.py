# adapters/dtls.py
import asyncio
import ssl
import struct
import time
import random
import os
import threading
from typing import List, Tuple

from .base import ProtocolAdapter
from .common import BENCH_HDR_FORMAT, BENCH_HDR_SIZE, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, TLS13_CIPHERS
from OpenSSL import SSL

# aiocoap related import
try:
    import aiocoap
    from aiocoap import Context, Message, Code
    AIOCOAP_AVAILABLE = True
except ImportError:
    AIOCOAP_AVAILABLE = False

# aiocoap.transports.dtls may not exist in all versions, guard the import
try:
    from aiocoap.transports.dtls import DtlsClientTransport, MessageInterface
    AIOCOAP_DTLS_AVAILABLE = True
except ImportError:
    AIOCOAP_DTLS_AVAILABLE = False

class DTLSAdapter(ProtocolAdapter):
    name = "dtls"

    def __init__(self):
        # Check dependencies
        if not AIOCOAP_DTLS_AVAILABLE:
            print("Warning: aiocoap.transports.dtls not available, using PyOpenSSL DTLS implementation")
        try:
            from OpenSSL import SSL
        except ImportError:
            raise RuntimeError("pyopenssl is not installed. Run 'pip install pyopenssl'")

        # Thread safety
        self.lock = threading.Lock()
        self.error_counter = {
            'rtt_results': [],
            'success_count': 0,
            'timeout_count': 0,
            'mismatch_count': 0,
            'short_response_count': 0,
            'unpack_error_count': 0,
            'handshake_error_count': 0,
            'other_error_count': 0,
        }
        self.stop_event = threading.Event()

    async def _run_async(self, host, port, cipher, size, rate, duration, warmup, ca):
        """PyOpenSSL + CoAP protocol direct implementation"""
        from socket import socket, AF_INET, SOCK_DGRAM
        import socket as sock_module
        import select

        # CoAP message creation function
        def make_coap_post(payload, msg_id):
            # CoAP POST message (confirmable, POST, path=/echo)
            # Ver=1, Type=CON(0), TKL=1
            ver_type_tkl = 0x41  # Ver=1, Type=0(CON), TKL=1
            code = 0x02  # POST
            mid_high = (msg_id >> 8) & 0xFF
            mid_low = msg_id & 0xFF
            token = 0x01
            coap_header = bytes([ver_type_tkl, code, mid_high, mid_low, token])
            # Option: Uri-Path = "echo" (option num 11, delta=11, len=4)
            coap_option = bytes([0xb4]) + b"echo"  # delta=11, len=4
            coap_end = bytes([0xff])  # payload marker
            return coap_header + coap_option + coap_end + payload

        # Resolve host
        try:
            resolved_host = sock_module.gethostbyname(host)
        except Exception as e:
            with self.lock:
                self.error_counter['other_error_count'] += 1
            return

        # UDP socket
        sock = socket(AF_INET, SOCK_DGRAM)
        sock.settimeout(5.0)
        sock.connect((resolved_host, port))

        # SSL Context setup
        ssl_context = SSL.Context(SSL.DTLS_METHOD)

        # Cipher setup
        if cipher == "aesgcm":
            cipher_str = "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:AES128-GCM-SHA256"
        elif cipher == "aes256gcm":
            cipher_str = "ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:AES256-GCM-SHA384:ECDHE-ECDSA-AES128-GCM-SHA256"
        elif cipher == "chacha20":
            cipher_str = "ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-GCM-SHA256"
        else:
            cipher_str = "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:AES128-GCM-SHA256"

        try:
            ssl_context.set_cipher_list(cipher_str.encode())
        except SSL.Error:
            cipher_str = "ALL:!aNULL:!eNULL"
            ssl_context.set_cipher_list(cipher_str.encode())

        # Disable certificate verification (only server needs certificate)
        ssl_context.set_verify(SSL.VERIFY_NONE, lambda conn, cert, errno, depth, ok: True)

        # DTLS connection
        conn = SSL.Connection(ssl_context, sock)
        conn.set_connect_state()

        try:
            # Handshake (non-blocking retry)
            handshake_start = time.monotonic()
            sock.setblocking(False)

            handshake_success = False
            while time.monotonic() - handshake_start < 10:
                try:
                    conn.do_handshake()
                    handshake_success = True
                    break
                except SSL.WantReadError:
                    readable, _, _ = select.select([sock], [], [], 0.5)
                    if not readable:
                        await asyncio.sleep(0.01)
                except SSL.WantWriteError:
                    _, writable, _ = select.select([], [sock], [], 0.5)
                    if not writable:
                        await asyncio.sleep(0.01)
                except SSL.Error as e:
                    with self.lock:
                        self.error_counter['handshake_error_count'] += 1
                    raise
                except Exception as e:
                    with self.lock:
                        self.error_counter['other_error_count'] += 1
                    raise

            if not handshake_success:
                with self.lock:
                    self.error_counter['handshake_error_count'] += 1
                raise TimeoutError("DTLS handshake timeout")

            # Now communicate with CoAP+DTLS
            period = 1.0 / rate
            start_time = time.monotonic()
            seq = 0
            payload_data = random.randbytes(size - BENCH_HDR_SIZE)

            while (elapsed := time.monotonic() - start_time) < duration:
                t_send_ns = time.monotonic_ns()
                header = struct.pack(BENCH_HDR_FORMAT, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, seq, t_send_ns, len(payload_data))
                packet = header + payload_data

                # Wrap in CoAP POST
                coap_msg = make_coap_post(packet, seq)

                try:
                    conn.send(coap_msg)

                    # Receive response (non-blocking)
                    resp = None
                    recv_timeout = 1.0
                    recv_start = time.monotonic()
                    while time.monotonic() - recv_start < recv_timeout:
                        try:
                            readable, _, _ = select.select([sock], [], [], 0.1)
                            if readable:
                                resp = conn.recv(4096)
                                if resp:
                                    break
                        except SSL.WantReadError:
                            await asyncio.sleep(0.01)
                        except Exception:
                            break

                    if resp and elapsed > warmup:
                        # Parse CoAP response (extract payload only)
                        # Payload is after 0xFF marker
                        if b'\xff' in resp:
                            resp_payload = resp.split(b'\xff', 1)[1]
                            if len(resp_payload) >= BENCH_HDR_SIZE:
                                try:
                                    resp_magic, _, resp_seq, _, _ = struct.unpack(BENCH_HDR_FORMAT, resp_payload[:BENCH_HDR_SIZE])
                                    if resp_magic == BENCH_HDR_MAGIC and resp_seq == seq:
                                        rtt_ns = time.monotonic_ns() - t_send_ns
                                        with self.lock:
                                            self.error_counter['rtt_results'].append((rtt_ns / 1e9, time.time()))
                                            self.error_counter['success_count'] += 1
                                    else:
                                        with self.lock:
                                            self.error_counter['mismatch_count'] += 1
                                except struct.error:
                                    with self.lock:
                                        self.error_counter['unpack_error_count'] += 1
                            else:
                                with self.lock:
                                    self.error_counter['short_response_count'] += 1
                    elif not resp and elapsed > warmup:
                        with self.lock:
                            self.error_counter['timeout_count'] += 1

                except Exception as e:
                    with self.lock:
                        self.error_counter['other_error_count'] += 1

                seq += 1
                await asyncio.sleep(period)

        except Exception as e:
            with self.lock:
                self.error_counter['other_error_count'] += 1
        finally:
            try:
                conn.shutdown()
            except:
                pass
            try:
                conn.close()
            except:
                pass
            sock.close()

    def run_load(self, host, port, cipher, size, rate, duration, warmup, ca=None, **kwargs):
        """
        Run load test using DTLS protocol.

        Returns:
            tuple: (thread, rtt_results_list) where rtt_results_list is thread-safe
        """
        # Reset counters
        with self.lock:
            self.error_counter['rtt_results'].clear()
            self.error_counter['success_count'] = 0
            self.error_counter['timeout_count'] = 0
            self.error_counter['mismatch_count'] = 0
            self.error_counter['short_response_count'] = 0
            self.error_counter['unpack_error_count'] = 0
            self.error_counter['handshake_error_count'] = 0
            self.error_counter['other_error_count'] = 0
        self.stop_event.clear()

        def thread_target():
            try:
                asyncio.run(self._run_async(host, port, cipher, size, rate, duration, warmup, ca))
            except Exception as e:
                print(f"DTLS adapter thread error: {e}")
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
                'mismatch_count': self.error_counter['mismatch_count'],
                'short_response_count': self.error_counter['short_response_count'],
                'unpack_error_count': self.error_counter['unpack_error_count'],
                'handshake_error_count': self.error_counter['handshake_error_count'],
                'other_error_count': self.error_counter['other_error_count'],
            }
