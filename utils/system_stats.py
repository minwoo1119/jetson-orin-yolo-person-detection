# utils/system_stats.py
import psutil
try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_GPU = True
except Exception:
    _GPU_HANDLE = None
    HAS_GPU = False

def sample_once():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    gpu = 0.0; gmem = 0.0
    if HAS_GPU and _GPU_HANDLE is not None:
        try:
            u = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE); gpu = float(u.gpu)
            m = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE); gmem = 100.0*m.used/m.total
        except Exception:
            pass
    return {"cpu_pct": cpu, "mem_pct": mem, "gpu_pct": gpu, "gpu_mem_pct": gmem}
