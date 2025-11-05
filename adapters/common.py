# adapters/common.py
import struct

# Packet header definition
BENCH_HDR_FORMAT = "!IHQQI"
BENCH_HDR_SIZE = struct.calcsize(BENCH_HDR_FORMAT)
BENCH_HDR_MAGIC = 0xDEADBEEF
BENCH_HDR_VERSION = 1

# TLS 1.3 Cipher Suite names
TLS13_CIPHERS = {
    "aesgcm": "TLS_AES_128_GCM_SHA256",
    "aes256gcm": "TLS_AES_256_GCM_SHA384",
    "chacha20": "TLS_CHACHA20_POLY1305_SHA256",
    "aesccm": "TLS_AES_128_CCM_SHA256",
    "aesccm8": "TLS_AES_128_CCM_8_SHA256",
}
