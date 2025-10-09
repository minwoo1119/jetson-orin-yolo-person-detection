# yolo_runner.py
import time
import threading
import psutil
import cv2
from ultralytics import YOLO
import numpy as np
import csv

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, pynvml.NVMLError):
    NVML_AVAILABLE = False

try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False

class YOLORunner:
    def __init__(self, model_path):
        import torch
        # Use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        if self.device == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"Using CPU")
        self.stats = {
            "time_sec": [],
            "cpu_pct": [],
            "mem_pct": [],
            "gpu_pct": [],
            "gpu_mem_pct": [],
            "interval_fps": [],
            "rtt_ms": [],
        }
        self.stop_event = threading.Event()
        self.frame_count_interval = 0
        self.rtt_data = None
        self.jtop_instance = None
        self.jtop_lock = threading.Lock()

    def _monitor_resources(self, process, duration_sec, per_second_writer=None):

        # Try to use jtop for Jetson GPU monitoring
        if JTOP_AVAILABLE and self.jtop_instance is None:
            try:
                self.jtop_instance = jtop()
                self.jtop_instance.start()
            except Exception as e:
                print(f"Failed to start jtop: {e}")
                self.jtop_instance = None

        for sec in range(duration_sec):
            if self.stop_event.is_set():
                break

            time.sleep(1.0) # 1초 대기

            cpu_pct = process.cpu_percent()
            mem_pct = process.memory_percent()

            self.stats["time_sec"].append(sec + 1)
            self.stats["cpu_pct"].append(cpu_pct)
            self.stats["mem_pct"].append(mem_pct)

            # Get GPU stats from jtop if available
            gpu_pct = 0
            gpu_mem_pct = 0
            if self.jtop_instance:
                try:
                    with self.jtop_lock:
                        if self.jtop_instance.ok():
                            gpu_pct = self.jtop_instance.stats.get('GPU', 0)
                            # Get GPU memory usage
                            gpu_mem = self.jtop_instance.memory.get('GPU', {})
                            if 'used' in gpu_mem and 'tot' in gpu_mem and gpu_mem['tot'] > 0:
                                gpu_mem_pct = (gpu_mem['used'] / gpu_mem['tot']) * 100
                except Exception:
                    pass

            self.stats["gpu_pct"].append(gpu_pct)
            self.stats["gpu_mem_pct"].append(gpu_mem_pct)

            self.stats["interval_fps"].append(self.frame_count_interval)

            # Calculate average RTT for this second from rtt_data
            current_time = time.time()
            rtt_ms = 0.0
            if self.rtt_data:
                recent_rtts = [rtt_sec * 1000 for rtt_sec, timestamp in self.rtt_data
                              if current_time - timestamp <= 1.0]
                rtt_ms = np.mean(recent_rtts) if recent_rtts else 0.0
            self.stats["rtt_ms"].append(rtt_ms)

            if per_second_writer:
                per_second_writer.writerow([sec + 1, cpu_pct, mem_pct, gpu_pct, gpu_mem_pct, self.frame_count_interval, rtt_ms])

            self.frame_count_interval = 0

    def run_video(self, video_path, duration_sec, rtt_data=None, overlay=False, per_second_writer=None):
        for key in self.stats:
            self.stats[key].clear()
        self.stop_event.clear()
        self.frame_count_interval = 0
        self.rtt_data = rtt_data

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        current_process = psutil.Process()
        # Call cpu_percent() once before the loop to initialize it
        current_process.cpu_percent()
        
        monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(current_process, duration_sec, per_second_writer),
            daemon=True
        )
        monitor_thread.start()

        start_time = time.time()
        total_frames = 0
        while time.time() - start_time < duration_sec:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            self.model.track(source=frame, persist=True, verbose=False, device=self.device)
            self.frame_count_interval += 1
            total_frames += 1

        self.stop_event.set()
        monitor_thread.join()
        cap.release()

        # Close jtop instance
        if self.jtop_instance:
            try:
                self.jtop_instance.close()
                self.jtop_instance = None
            except Exception:
                pass

        actual_duration = time.time() - start_time
        avg_fps = total_frames / actual_duration if actual_duration > 0 else 0

        summary = {
            "start_time": start_time,
            "avg_fps": avg_fps,
            "cpu_pct": np.mean(self.stats["cpu_pct"]) if self.stats["cpu_pct"] else 0,
            "gpu_pct": np.mean(self.stats["gpu_pct"]) if self.stats["gpu_pct"] else 0,
            "mem_pct": np.mean(self.stats["mem_pct"]) if self.stats["mem_pct"] else 0,
            "gpu_mem_pct": np.mean(self.stats["gpu_mem_pct"]) if self.stats["gpu_mem_pct"] else 0,
        }

        return summary

if NVML_AVAILABLE:
    # This should be called when the application exits.
    # A better design would be to manage this in the main script.
    pass