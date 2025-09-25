# run_suite.py
import argparse, os, time, csv, sys, yaml, threading
import numpy as np
from yolo_runner import YOLORunner

def load_adapter(proto):
    if proto == "https":
        from adapters.https_tls import HTTPSAdapter
        return HTTPSAdapter()
    if proto == "dtls":
        from adapters.dtls import DTLSAdapter
        return DTلسAdapter()
    if proto == "mqtt":
        from adapters.mqtt_tls import MQTTAdapter
        return MQTTAdapter()
    if proto == "http3":
        from adapters.http3_quic import HTTP3Adapter
        return HTTP3Adapter()
    if proto == "coap":
        from adapters.coap_oscore import CoAPAdapter
        return CoAPAdapter()
    raise ValueError(f"Unknown proto: {proto}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="out_results.csv")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    board = cfg.get("board", "unknown")
    duration = int(cfg.get("duration_sec", 30))
    warmup = int(cfg.get("warmup_sec", 5))
    size = int(cfg.get("payload_size", 256))
    rate = int(cfg.get("send_rate", 50))
    endpoints = cfg.get("bench_server", {})
    matrix = cfg.get("matrix", {})
    yolo_cfg = cfg.get("yolo", {})
    model_path = yolo_cfg.get("model_path", "yolo_model/yolov8n.pt")
    videos = yolo_cfg.get("videos", {})

    if not videos:
        print("No videos configured under yolo.videos", file=sys.stderr)
        sys.exit(1)

    yolo = YOLORunner(model_path)

    with open(args.out, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["board","video","proto","cipher",
                    "avg_fps","cpu_pct","gpu_pct","mem_pct","gpu_mem_pct",
                    "rtt_p50_ms","rtt_p95_ms","rtt_p99_ms","note"])

        for video_name, video_path in videos.items():
            for proto, ciphers in matrix.items():
                for cipher in ciphers:
                    ep = endpoints.get(proto, {})
                    host, port = ep.get("host"), ep.get("port")
                    ca, cert, key = ep.get("ca"), ep.get("cert"), ep.get("key")
                    oscore_context = ep.get("oscore_context")

                    print(f"\n=== RUN {board} | {video_name} | {proto} | {cipher} ===")
                    try:
                        adapter = load_adapter(proto)
                    except Exception as e:
                        note = f"adapter load fail: {e}"
                        print(f"[SKIP] {note}")
                        w.writerow([board, video_name, proto, cipher, 0,0,0,0,0, 0,0,0, note]); out.flush()
                        continue

                    handle, lines = None, []
                    try:
                        handle, lines = adapter.run_load(
                            host=host, port=port, cipher=cipher,
                            size=size, rate=rate, duration=duration, warmup=warmup,
                            ca=ca, cert=cert, key=key, oscore_context=oscore_context
                        )
                    except Exception as e:
                        note = f"net load fail: {e}"
                        print(f"[SKIP] {note}")
                        w.writerow([board, video_name, proto, cipher, 0,0,0,0,0, 0,0,0, note]); out.flush()
                        continue

                    # 화면에 영상을 그리지 않고 백그라운드로 실행
                    yres = yolo.run_video(video_path, duration_sec=duration, overlay=True)

                    if isinstance(handle, threading.Thread):
                        handle.join(timeout=10) # Wait for network thread to finish

                    rtt_p50, rtt_p95, rtt_p99 = 0, 0, 0
                    note = ""
                    if lines and isinstance(lines, list) and len(lines) > 0:
                        rtt_ms = [r * 1000 for r in lines]
                        rtt_p50 = np.percentile(rtt_ms, 50)
                        rtt_p95 = np.percentile(rtt_ms, 95)
                        rtt_p99 = np.percentile(rtt_ms, 99)
                    else:
                        note = "no rtt results"

                    w.writerow([board, video_name, proto, cipher,
                                f"{yres['avg_fps']:.3f}", f"{yres['cpu_pct']:.2f}",
                                f"{yres['gpu_pct']:.2f}", f"{yres['mem_pct']:.2f}",
                                f"{yres['gpu_mem_pct']:.2f}", f"{rtt_p50:.3f}",
                                f"{rtt_p95:.3f}", f"{rtt_p99:.3f}", note])
                    out.flush()
                    print(f"=== DONE | FPS {yres['avg_fps']:.2f} | RTT p50 {rtt_p50:.2f}ms ===")

    print(f"\n결과 CSV: {args.out}")

if __name__ == "__main__":
    main()