# run_suite.py
import argparse, os, csv, sys, yaml, threading
from datetime import datetime
import numpy as np
from yolo_runner import YOLORunner

def load_adapter(proto):
    if proto == "https":
        from adapters.https_tls import HTTPSAdapter
        return HTTPSAdapter()
    if proto == "dtls":
        from adapters.dtls import DTLSAdapter
        return DTLSAdapter()
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
    experiment_label = cfg.get("experiment_label", "").strip()
    duration = int(cfg.get("duration_sec", 30))
    warmup = int(cfg.get("warmup_sec", 5))

    # Create timestamped result directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_label:
        result_dir = f"results/{timestamp}_{experiment_label}"
    else:
        result_dir = f"results/{timestamp}"

    os.makedirs(result_dir, exist_ok=True)
    timeseries_dir = os.path.join(result_dir, "timeseries_logs")
    plots_dir = os.path.join(result_dir, "plots")
    os.makedirs(timeseries_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Save a copy of the config file for reference
    import shutil
    config_backup = os.path.join(result_dir, "suite.yaml")
    shutil.copy(args.config, config_backup)

    print(f"Results will be saved to: {result_dir}")

    # Support both single value and list for payload_sizes and send_rates
    payload_sizes = cfg.get("payload_sizes")
    if payload_sizes is None:
        # Backward compatibility: try old single value
        payload_sizes = [int(cfg.get("payload_size", 256))]
    elif isinstance(payload_sizes, int):
        payload_sizes = [payload_sizes]
    else:
        payload_sizes = [int(s) for s in payload_sizes]

    send_rates = cfg.get("send_rates")
    if send_rates is None:
        # Backward compatibility: try old single value
        send_rates = [int(cfg.get("send_rate", 50))]
    elif isinstance(send_rates, int):
        send_rates = [send_rates]
    else:
        send_rates = [int(r) for r in send_rates]

    endpoints = cfg.get("bench_server", {})
    matrix = cfg.get("matrix", {})
    yolo_cfg = cfg.get("yolo", {})
    model_path = yolo_cfg.get("model_path", "yolo_model/yolov8n.pt")
    num_models = int(yolo_cfg.get("num_models", 1))  # 기본값 1
    videos = yolo_cfg.get("videos", {})

    if not videos:
        print("No videos configured under yolo.videos", file=sys.stderr)
        sys.exit(1)

    yolo = YOLORunner(model_path)

    # Determine output CSV path (use result_dir if default)
    if args.out == "out_results.csv":
        out_csv_path = os.path.join(result_dir, "results.csv")
    else:
        out_csv_path = args.out

    with open(out_csv_path, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["board","video","proto","cipher","payload_size","send_rate",
                    "avg_fps","cpu_pct","gpu_pct","mem_pct","gpu_mem_pct",
                    "rtt_p50_ms","rtt_p95_ms","rtt_p99_ms","note"])

        for video_name, video_path in videos.items():
            for proto, ciphers in matrix.items():
                for cipher in ciphers:
                    for size in payload_sizes:
                        for rate in send_rates:
                            ep = endpoints.get(proto, {})
                            host, port = ep.get("host"), ep.get("port")
                            ca, cert, key = ep.get("ca"), ep.get("cert"), ep.get("key")
                            oscore_context = ep.get("oscore_context")

                            print(f"\n=== RUN {board} | {video_name} | {proto} | {cipher} | size={size} | rate={rate} ===")
                            adapter = None
                            handle, lines = None, []

                            log_filename = os.path.join(timeseries_dir, f"{video_name}_{proto}_{cipher}_size{size}_rate{rate}.csv")
                            try:
                                with open(log_filename, "w", newline="", encoding="utf-8") as per_second_out_file:
                                    per_second_writer = csv.writer(per_second_out_file)
                                    per_second_writer.writerow(["time_sec","cpu_pct","mem_pct","gpu_pct","gpu_mem_pct","interval_fps","rtt_ms"])

                                    adapter = load_adapter(proto)
                                    handle, lines = adapter.run_load(
                                        host=host, port=port, cipher=cipher,
                                        size=size, rate=rate, duration=duration, warmup=warmup,
                                        ca=ca, cert=cert, key=key, oscore_context=oscore_context
                                    )

                                    # Get the lock from adapter for thread-safe RTT data access
                                    rtt_data_lock = getattr(adapter, 'lock', None)

                                    # Pass RTT data and lock to YOLORunner (CRITICAL FIX!)
                                    yres = yolo.run_video(
                                        video_path,
                                        duration_sec=duration,
                                        warmup_sec=warmup,
                                        rtt_data=lines,
                                        rtt_data_lock=rtt_data_lock,
                                        overlay=False,
                                        per_second_writer=per_second_writer,
                                        num_models=num_models
                                    )

                                    # Wait for network thread to finish (with proper timeout)
                                    if isinstance(handle, threading.Thread):
                                        handle.join(timeout=duration + 10)
                                        if handle.is_alive():
                                            print(f"Warning: Network thread still running after {duration + 10}s timeout")

                                    if hasattr(adapter, 'stop_load'):
                                        adapter.stop_load()

                                    # Calculate RTT percentiles with thread-safe access
                                    rtt_p50, rtt_p95, rtt_p99 = 0, 0, 0
                                    note = ""
                                    if lines and isinstance(lines, list) and len(lines) > 0:
                                        # Thread-safe read of RTT results
                                        if rtt_data_lock:
                                            with rtt_data_lock:
                                                rtt_ms = [r[0] * 1000 if isinstance(r, tuple) else r * 1000 for r in lines]
                                        else:
                                            rtt_ms = [r[0] * 1000 if isinstance(r, tuple) else r * 1000 for r in lines]

                                        if len(rtt_ms) > 0:
                                            rtt_p50 = np.percentile(rtt_ms, 50)
                                            rtt_p95 = np.percentile(rtt_ms, 95)
                                            rtt_p99 = np.percentile(rtt_ms, 99)
                                        else:
                                            note = "no rtt results"
                                    else:
                                        note = "no rtt results"

                                    # Collect error statistics for logging
                                    if hasattr(adapter, 'get_error_stats'):
                                        error_stats = adapter.get_error_stats()
                                        if error_stats.get('timeout_count', 0) > 0 or error_stats.get('request_error_count', 0) > 0:
                                            print(f"Warning: Network errors detected: {error_stats}")

                                    w.writerow([board, video_name, proto, cipher, size, rate,
                                                f"{yres['avg_fps']:.3f}", f"{yres['cpu_pct']:.2f}",
                                                f"{yres['gpu_pct']:.2f}", f"{yres['mem_pct']:.2f}",
                                                f"{yres['gpu_mem_pct']:.2f}", f"{rtt_p50:.3f}",
                                                f"{rtt_p95:.3f}", f"{rtt_p99:.3f}", note])
                                    out.flush()
                                    print(f"=== DONE | FPS {yres['avg_fps']:.2f} | RTT p50 {rtt_p50:.2f}ms ===")

                            except Exception as e:
                                note = f"run failed: {e}"
                                print(f"[FAIL] {note}")
                                import traceback
                                traceback.print_exc()
                                w.writerow([board, video_name, proto, cipher, size, rate, *([0]*9), note]); out.flush()
                                if isinstance(handle, threading.Thread):
                                    handle.join(timeout=1)
                                if adapter and hasattr(adapter, 'stop_load'):
                                    adapter.stop_load()
                                continue

    print(f"\n" + "="*60)
    print(f"실험 완료!")
    print(f"결과 디렉토리: {result_dir}")
    print(f"요약 CSV: {out_csv_path}")
    print(f"타임시리즈 로그: {timeseries_dir}")
    print(f"="*60)

if __name__ == "__main__":
    main()
