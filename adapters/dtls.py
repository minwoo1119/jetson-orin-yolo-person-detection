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
from OpenSSL import SSL

# aiocoap 관련 import
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
    """Jetson 원본과 동일"""
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
        # Jetson과 동일: aiocoap DTLS transport 확인
        if not AIOCOAP_DTLS_AVAILABLE:
            # 라즈베리파이용: 실제로 없으므로 warning만 출력하고 계속 진행
            print("Warning: aiocoap.transports.dtls not available, attempting alternative DTLS implementation")
        try:
            from OpenSSL import SSL
        except ImportError:
            raise RuntimeError("pyopenssl is not installed. Run 'pip install pyopenssl'")

    async def _coap_client_sender(self, host, port, cipher, size, rate, duration, warmup, ca):
        """CoAP+DTLS를 사용하는 클라이언트 (서버 인증서 검증 없음)"""
        if not AIOCOAP_AVAILABLE:
            raise RuntimeError("aiocoap is not installed")

        rtt_results = []
        period = 1.0 / rate
        start_time = time.monotonic()
        seq = 0
        payload_data = random.randbytes(size - BENCH_HDR_SIZE)

        # CoAP Context 생성
        # tinydtls 백엔드는 PSK만 지원하므로, 임시로 빈 PSK 설정 시도
        # 또는 credential 없이 시도 (서버가 client auth를 NONE으로 설정했으므로)
        try:
            from aiocoap.credentials import CredentialsMap
            credentials = CredentialsMap()
            # 빈 credentials로 시도
            context = await Context.create_client_context()
            context.client_credentials = credentials
        except Exception as e:
            # credential 설정 실패 시 기본 context 사용
            print(f"Could not set credentials: {e}")
            context = await Context.create_client_context()

        # coaps:// URI 생성
        uri = f"coaps://{host}:{port}/echo"

        print(f"CoAP+DTLS connecting to {uri} with cipher preference: {cipher}")

        try:
            while (elapsed := time.monotonic() - start_time) < duration:
                t_send_ns = time.monotonic_ns()
                header = struct.pack(BENCH_HDR_FORMAT, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, seq, t_send_ns, len(payload_data))
                packet = header + payload_data

                try:
                    # CoAP POST 요청 생성
                    request = Message(code=Code.POST, uri=uri, payload=packet)

                    # 요청 전송 및 응답 대기 (타임아웃 2초)
                    response = await asyncio.wait_for(context.request(request).response, timeout=2.0)

                    if elapsed > warmup and response.payload:
                        resp_payload = response.payload
                        if len(resp_payload) >= BENCH_HDR_SIZE:
                            try:
                                resp_magic, _, resp_seq, _, _ = struct.unpack(BENCH_HDR_FORMAT, resp_payload[:BENCH_HDR_SIZE])
                                if resp_magic == BENCH_HDR_MAGIC and resp_seq == seq:
                                    rtt_ns = time.monotonic_ns() - t_send_ns
                                    rtt_results.append((rtt_ns / 1e9, time.monotonic()))
                            except struct.error:
                                # 패킷 형식 불일치
                                pass
                except asyncio.TimeoutError:
                    # 타임아웃은 정상 (응답 없을 수 있음)
                    pass
                except Exception as e:
                    # 첫 요청에서 credential 에러가 나면 상세히 출력
                    if seq < 5 and "credential" in str(e).lower():
                        print(f"CoAP DTLS credential error: {e}")
                    # 그 외 에러는 조용히 무시

                seq += 1
                await asyncio.sleep(period)
        finally:
            await context.shutdown()

        return rtt_results

    async def _run_async(self, host, port, cipher, size, rate, duration, warmup, ca):
        """PyOpenSSL + CoAP 프로토콜 직접 구현"""
        from socket import socket, AF_INET, SOCK_DGRAM
        import socket as sock_module

        # CoAP 메시지 생성 함수
        def make_coap_post(payload, msg_id):
            # CoAP POST 메시지 (confirmable, POST, path=/echo)
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

        # 호스트 해석
        try:
            resolved_host = sock_module.gethostbyname(host)
            print(f"DTLS connecting to {resolved_host}:{port} (cipher: {cipher})")
        except Exception as e:
            print(f"Failed to resolve {host}: {e}")
            return []

        # UDP 소켓
        sock = socket(AF_INET, SOCK_DGRAM)
        sock.settimeout(5.0)
        sock.connect((resolved_host, port))

        # SSL Context 설정
        ssl_context = SSL.Context(SSL.DTLS_METHOD)

        # Cipher 설정 (서버가 지원하는 cipher로 fallback 추가)
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

        # 인증서 검증 비활성화 (서버만 인증서 필요)
        ssl_context.set_verify(SSL.VERIFY_NONE, lambda conn, cert, errno, depth, ok: True)

        # DTLS 연결
        conn = SSL.Connection(ssl_context, sock)
        conn.set_connect_state()

        try:
            # 핸드셰이크 (논블로킹 retry)
            import select
            print(f"[DEBUG] Starting DTLS handshake...")
            handshake_start = time.monotonic()
            sock.setblocking(False)  # 논블로킹 모드로 변경

            handshake_attempts = 0
            want_read_count = 0
            want_write_count = 0

            while time.monotonic() - handshake_start < 10:
                try:
                    handshake_attempts += 1
                    conn.do_handshake()
                    actual_cipher = conn.get_cipher_name()
                    print(f"[SUCCESS] DTLS handshake OK with cipher: {actual_cipher}")
                    print(f"[DEBUG] Handshake stats: attempts={handshake_attempts}, want_read={want_read_count}, want_write={want_write_count}")
                    break
                except SSL.WantReadError:
                    want_read_count += 1
                    readable, _, _ = select.select([sock], [], [], 0.5)
                    if not readable:
                        if handshake_attempts <= 5:
                            print(f"[DEBUG] WantReadError: no data available (attempt {handshake_attempts})")
                        await asyncio.sleep(0.01)
                except SSL.WantWriteError:
                    want_write_count += 1
                    _, writable, _ = select.select([], [sock], [], 0.5)
                    if not writable:
                        if handshake_attempts <= 5:
                            print(f"[DEBUG] WantWriteError: socket not writable (attempt {handshake_attempts})")
                        await asyncio.sleep(0.01)
                except SSL.Error as e:
                    print(f"[ERROR] SSL Error during handshake: {e}")
                    raise
                except Exception as e:
                    print(f"[ERROR] Unexpected error during handshake: {type(e).__name__}: {e}")
                    raise
            else:
                elapsed = time.monotonic() - handshake_start
                print(f"[FAIL] DTLS handshake timeout after {elapsed:.2f}s")
                print(f"[DEBUG] Final stats: attempts={handshake_attempts}, want_read={want_read_count}, want_write={want_write_count}")
                raise TimeoutError("DTLS handshake timeout")

            # 이제 CoAP+DTLS로 통신
            rtt_results = []
            period = 1.0 / rate
            start_time = time.monotonic()
            seq = 0
            payload_data = random.randbytes(size - BENCH_HDR_SIZE)

            while (elapsed := time.monotonic() - start_time) < duration:
                t_send_ns = time.monotonic_ns()
                header = struct.pack(BENCH_HDR_FORMAT, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, seq, t_send_ns, len(payload_data))
                packet = header + payload_data

                # CoAP POST로 감싸기
                coap_msg = make_coap_post(packet, seq)

                try:
                    conn.send(coap_msg)

                    # 응답 수신 (논블로킹)
                    import select
                    resp = None
                    recv_timeout = 1.0  # 1초 타임아웃
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
                        # CoAP 응답 파싱 (payload만 추출)
                        # 0xFF 마커 이후가 payload
                        if b'\xff' in resp:
                            resp_payload = resp.split(b'\xff', 1)[1]
                            if len(resp_payload) >= BENCH_HDR_SIZE:
                                try:
                                    resp_magic, _, resp_seq, _, _ = struct.unpack(BENCH_HDR_FORMAT, resp_payload[:BENCH_HDR_SIZE])
                                    if resp_magic == BENCH_HDR_MAGIC and resp_seq == seq:
                                        rtt_ns = time.monotonic_ns() - t_send_ns
                                        rtt_results.append((rtt_ns / 1e9, time.monotonic()))
                                        if len(rtt_results) <= 3:
                                            print(f"Got valid response {len(rtt_results)}, RTT: {rtt_ns/1e6:.2f}ms")
                                except struct.error:
                                    pass
                except Exception as e:
                    if seq < 3:
                        print(f"DTLS send/recv error: {e}")

                seq += 1
                await asyncio.sleep(period)

            return rtt_results

        except Exception as e:
            import traceback
            print(f"DTLS connection failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            return []
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

        # 아래 코드는 사용하지 않음 (삭제 예정)
        ssl_context = SSL.Context(SSL.DTLS_METHOD)

        # DTLS 1.2 호환 cipher suites
        # 서버의 Californium 설정에 맞춰 cipher 선택
        # 서버는 ECDHE 기반 cipher suite를 주로 지원
        if cipher == "aesgcm":
            # AES128-GCM 우선
            cipher_str = "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:AES128-GCM-SHA256"
        elif cipher == "aes256gcm":
            # AES256-GCM 우선
            cipher_str = "ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:AES256-GCM-SHA384"
        elif cipher == "chacha20":
            # ChaCha20-Poly1305 우선
            cipher_str = "ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305"
        else:
            # 기본값: 모든 안전한 cipher 허용
            cipher_str = "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384"

        # Fallback cipher 추가 (호환성 향상)
        cipher_str += ":HIGH:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4"

        try:
            ssl_context.set_cipher_list(cipher_str.encode())
        except SSL.Error as e:
            print(f"Warning: Failed to set cipher {cipher_str}: {e}")
            # 최종 Fallback
            try:
                ssl_context.set_cipher_list(b"ALL:!aNULL:!eNULL")
            except SSL.Error as e2:
                print(f"Error: Cannot set any ciphers: {e2}")

        if ca and os.path.exists(ca):
            ssl_context.load_verify_locations(cafile=ca)
            ssl_context.set_verify(SSL.VERIFY_PEER, lambda conn, cert, errno, depth, ok: ok)
        else:
            ssl_context.set_verify(SSL.VERIFY_NONE, lambda conn, cert, errno, depth, ok: True)

        # Jetson에서는 DtlsClientTransport를 사용하지만 라즈베리파이에는 없음
        # 대신 직접 UDP + DTLS 구현
        if AIOCOAP_DTLS_AVAILABLE:
            # Jetson 원본 방식
            transport, session = None, None
            try:
                transport = await DtlsClientTransport.create_client_transport_endpoint(ssl_context=ssl_context)
                session = await transport.get_message_interface((host, port))
                return await client_sender(session, size, rate, duration, warmup)
            finally:
                if session: session.shutdown()
                if transport: await transport.shutdown()
        else:
            # 라즈베리파이 대체 구현: 동기 방식으로 실행
            from socket import socket, AF_INET, SOCK_DGRAM
            import socket as sock_module
            import select

            # 호스트 해석
            try:
                resolved_host = sock_module.gethostbyname(host)
            except Exception as e:
                print(f"Failed to resolve host {host}: {e}")
                resolved_host = host

            # UDP 소켓
            sock = socket(AF_INET, SOCK_DGRAM)
            sock.setblocking(False)  # 논블로킹 모드로 설정
            sock.connect((resolved_host, port))

            # DTLS 연결
            conn = SSL.Connection(ssl_context, sock)
            conn.set_connect_state()

            try:
                # 핸드셰이크 (타임아웃 15초)
                handshake_start = time.monotonic()
                handshake_done = False

                while not handshake_done and (time.monotonic() - handshake_start < 15):
                    try:
                        conn.do_handshake()
                        # 성공한 cipher 확인
                        actual_cipher = conn.get_cipher_name()
                        print(f"DTLS handshake successful with {resolved_host}:{port} using cipher: {actual_cipher}")
                        handshake_done = True
                    except SSL.WantReadError:
                        # select로 읽기 가능할 때까지 대기
                        readable, _, _ = select.select([sock], [], [], 0.5)
                        if not readable:
                            await asyncio.sleep(0.01)
                    except SSL.WantWriteError:
                        # select로 쓰기 가능할 때까지 대기
                        _, writable, _ = select.select([], [sock], [], 0.5)
                        if not writable:
                            await asyncio.sleep(0.01)
                    except (SSL.Error, SSL.SysCallError) as e:
                        error_msg = str(e)
                        print(f"DTLS handshake failed with cipher config '{cipher}': {error_msg}")
                        raise

                if not handshake_done:
                    raise TimeoutError(f"DTLS handshake timeout after 15 seconds")

                # 동기 client_sender를 비동기로 래핑
                return await self._client_sender_sync_wrapper(conn, size, rate, duration, warmup)
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

    async def _client_sender_sync_wrapper(self, conn, size, rate, duration, warmup):
        """동기 DTLS 연결을 비동기로 래핑"""
        import select
        rtt_results = []
        period = 1.0 / rate
        start_time = time.monotonic()
        seq = 0
        payload_data = random.randbytes(size - BENCH_HDR_SIZE)

        # 소켓 가져오기
        sock = conn.fileno()

        while (elapsed := time.monotonic() - start_time) < duration:
            t_send_ns = time.monotonic_ns()
            header = struct.pack(BENCH_HDR_FORMAT, BENCH_HDR_MAGIC, BENCH_HDR_VERSION, seq, t_send_ns, len(payload_data))
            packet = header + payload_data

            try:
                # 송신 (논블로킹)
                while True:
                    try:
                        conn.send(packet)
                        break
                    except SSL.WantWriteError:
                        _, writable, _ = select.select([], [sock], [], 0.5)
                        if not writable:
                            await asyncio.sleep(0.01)
                    except SSL.WantReadError:
                        readable, _, _ = select.select([sock], [], [], 0.5)
                        if not readable:
                            await asyncio.sleep(0.01)

                # 수신 (논블로킹, 타임아웃 2초)
                recv_start = time.monotonic()
                resp_payload = None
                while (time.monotonic() - recv_start < 2.0) and not resp_payload:
                    try:
                        resp_payload = conn.recv(4096)
                    except SSL.WantReadError:
                        readable, _, _ = select.select([sock], [], [], 0.1)
                        if not readable:
                            await asyncio.sleep(0.01)
                    except SSL.WantWriteError:
                        _, writable, _ = select.select([], [sock], [], 0.1)
                        if not writable:
                            await asyncio.sleep(0.01)
                    except SSL.ZeroReturnError:
                        # 연결 종료
                        break
                    except Exception:
                        # 기타 에러
                        break

                # RTT 측정 (warmup 관계없이 응답 검증)
                if resp_payload and len(resp_payload) >= BENCH_HDR_SIZE:
                    try:
                        resp_magic, _, resp_seq, _, _ = struct.unpack(BENCH_HDR_FORMAT, resp_payload[:BENCH_HDR_SIZE])
                        if resp_magic == BENCH_HDR_MAGIC and resp_seq == seq:
                            rtt_ns = time.monotonic_ns() - t_send_ns
                            if elapsed > warmup:  # warmup 이후만 기록
                                rtt_results.append((rtt_ns / 1e9, time.monotonic()))
                    except struct.error:
                        # 패킷 형식이 맞지 않으면 무시
                        pass
            except Exception as e:
                if str(e):  # 빈 에러 메시지 제외
                    print(f"DTLS request failed: {e}")

            seq += 1
            await asyncio.sleep(period)

        return rtt_results

    def run_load(self, host, port, cipher, size, rate, duration, warmup, ca=None, **kwargs):
        """Jetson 원본과 완전히 동일"""
        results_list = []
        def thread_target():
            rtt_results = asyncio.run(self._run_async(host, port, cipher, FIXED_PACKET_SIZE, rate, duration, warmup, ca))
            results_list.extend(rtt_results)

        thread = threading.Thread(target=thread_target, daemon=True)
        thread.start()
        return thread, results_list
