# yolo_runner.py
import time
import threading
import psutil
import cv2
import os
from ultralytics import YOLO

# pynvml을 사용하되, 설치되지 않았을 경우를 대비한 예외 처리
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, pynvml.NVMLError):
    NVML_AVAILABLE = False

class YOLORunner:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.stats = {
            "cpu_pct": [], "mem_pct": [],
            "gpu_pct": [], "gpu_mem_pct": [],
        }
        self.stop_event = threading.Event()

    def _monitor_resources(self, process, duration_sec):
        start_time = time.time()
        gpu_handle = None
        if NVML_AVAILABLE:
            try:
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except pynvml.NVMLError:
                gpu_handle = None

        while not self.stop_event.is_set() and (time.time() - start_time) < duration_sec:
            self.stats["cpu_pct"].append(process.cpu_percent())
            self.stats["mem_pct"].append(process.memory_percent())
            if gpu_handle:
                try:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    self.stats["gpu_pct"].append(gpu_util.gpu)
                    self.stats["gpu_mem_pct"].append(100 * gpu_mem.used / gpu_mem.total)
                except pynvml.NVMLError:
                    self.stats["gpu_pct"].append(0)
                    self.stats["gpu_mem_pct"].append(0)
            else:
                self.stats["gpu_pct"].append(0)
                self.stats["gpu_mem_pct"].append(0)
            time.sleep(0.5)

    def run_video(self, video_path, duration_sec, overlay=False):
        for key in self.stats: self.stats[key].clear()
        self.stop_event.clear()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        current_process = psutil.Process()
        monitor_thread = threading.Thread(target=self._monitor_resources, args=(current_process, duration_sec), daemon=True)
        monitor_thread.start()

        frame_count = 0
        start_time = time.time()
        window_name = "YOLO Benchmark"

        if overlay:
            # Check for display environment
            if 'DISPLAY' not in os.environ:
                print("WARNING: No display environment found. Video will not be shown.")
                overlay = False
            else:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration_sec:
                break

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Run tracking, get results
            results = self.model.track(source=frame, persist=True, verbose=False)
            frame_count += 1

            if overlay and results:
                # Get the annotated frame and display it in our own window
                annotated_frame = results[0].plot()
                cv2.imshow(window_name, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        self.stop_event.set()
        monitor_thread.join()
        cap.release()
        if overlay: cv2.destroyAllWindows()

        actual_duration = time.time() - start_time
        avg_fps = frame_count / actual_duration if actual_duration > 0 else 0
        
        avg_stats = {}
        for key, values in self.stats.items():
            avg_stats[key] = sum(values) / len(values) if values else 0

        return {
            "avg_fps": avg_fps,
            "cpu_pct": avg_stats["cpu_pct"],
            "gpu_pct": avg_stats["gpu_pct"],
            "mem_pct": avg_stats["mem_pct"],
            "gpu_mem_pct": avg_stats["gpu_mem_pct"],
        }

if NVML_AVAILABLE:
    pynvml.nvmlShutdown()