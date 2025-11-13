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
        self.model_path = model_path
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
        self.frame_count_lock = threading.Lock()  # 프레임 카운트 동기화용
        self.rtt_data = None
        self.jtop_instance = None
        self.jtop_lock = threading.Lock()

    def _monitor_resources(self, process, duration_sec, warmup_sec, per_second_writer=None, rtt_data_lock=None):
        """
        Monitor system resources every second.

        Args:
            process: psutil.Process to monitor
            duration_sec: Total duration including warmup
            warmup_sec: Warmup period to skip before collecting RTT
            per_second_writer: CSV writer for per-second logs
            rtt_data_lock: Lock for accessing rtt_data safely
        """
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

            time.sleep(1.0)  # 1초 대기

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

            # 스레드 안전하게 프레임 카운트 읽기 및 리셋
            with self.frame_count_lock:
                interval_fps = self.frame_count_interval
                self.frame_count_interval = 0

            self.stats["interval_fps"].append(interval_fps)

            # Calculate average RTT for this second from rtt_data
            # Only collect RTT after warmup period
            rtt_ms = 0.0
            if sec >= warmup_sec and self.rtt_data is not None:
                current_time = time.time()
                # Thread-safe read of rtt_data
                if rtt_data_lock:
                    with rtt_data_lock:
                        recent_rtts = [rtt_sec * 1000 for rtt_sec, timestamp in self.rtt_data
                                      if current_time - timestamp <= 1.0]
                else:
                    # Fallback if no lock provided (for backward compatibility)
                    recent_rtts = [rtt_sec * 1000 for rtt_sec, timestamp in self.rtt_data
                                  if current_time - timestamp <= 1.0]
                rtt_ms = np.mean(recent_rtts) if recent_rtts else 0.0
            self.stats["rtt_ms"].append(rtt_ms)

            if per_second_writer:
                per_second_writer.writerow([sec + 1, cpu_pct, mem_pct, gpu_pct, gpu_mem_pct, interval_fps, rtt_ms])

    def _worker_thread(self, video_path, duration_sec, start_time):
        """각 모델 인스턴스가 독립적으로 비디오를 처리하는 워커 스레드"""
        # 각 스레드마다 독립적인 YOLO 모델 인스턴스 생성
        model = YOLO(self.model_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Worker thread cannot open video file: {video_path}")
            return

        thread_total_frames = 0
        while time.time() - start_time < duration_sec and not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # YOLO 추론 실행
            model.track(source=frame, persist=True, verbose=False, device=self.device)

            # 스레드 안전하게 프레임 카운트 증가
            with self.frame_count_lock:
                self.frame_count_interval += 1

            thread_total_frames += 1

        cap.release()

    def run_video(self, video_path, duration_sec, warmup_sec=0, rtt_data=None, rtt_data_lock=None, overlay=False, per_second_writer=None, num_models=1):
        """
        Run YOLO inference on video while monitoring resources.

        Args:
            video_path: Path to video file
            duration_sec: Total duration (including warmup)
            warmup_sec: Warmup period before collecting RTT (default: 0)
            rtt_data: List of (rtt_seconds, timestamp) tuples from network adapter
            rtt_data_lock: Threading lock for safe access to rtt_data
            overlay: Whether to display video (deprecated, kept for compatibility)
            per_second_writer: CSV writer for per-second statistics
            num_models: Number of YOLO model instances to run in parallel

        Returns:
            dict: Summary statistics (avg_fps, cpu_pct, gpu_pct, mem_pct, gpu_mem_pct)
        """
        for key in self.stats:
            self.stats[key].clear()
        self.stop_event.clear()
        self.frame_count_interval = 0
        self.rtt_data = rtt_data

        # 비디오 파일이 유효한지 확인
        test_cap = cv2.VideoCapture(video_path)
        if not test_cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        test_cap.release()

        current_process = psutil.Process()
        # Call cpu_percent() once before the loop to initialize it
        current_process.cpu_percent()

        monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(current_process, duration_sec, warmup_sec, per_second_writer, rtt_data_lock),
            daemon=True
        )
        monitor_thread.start()

        start_time = time.time()

        # 여러 워커 스레드 생성 및 실행
        worker_threads = []
        print(f"Starting {num_models} YOLO model instances...")
        for i in range(num_models):
            worker = threading.Thread(
                target=self._worker_thread,
                args=(video_path, duration_sec, start_time),
                daemon=True
            )
            worker.start()
            worker_threads.append(worker)

        # 모든 워커 스레드가 종료될 때까지 대기
        for worker in worker_threads:
            worker.join()

        self.stop_event.set()
        monitor_thread.join()

        # Close jtop instance
        if self.jtop_instance:
            try:
                self.jtop_instance.close()
                self.jtop_instance = None
            except Exception:
                pass

        actual_duration = time.time() - start_time
        # 모든 워커 스레드가 처리한 총 프레임 수 계산
        total_frames = sum(self.stats["interval_fps"]) if self.stats["interval_fps"] else 0
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