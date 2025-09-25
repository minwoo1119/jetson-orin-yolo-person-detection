# adapters/base.py
import subprocess, shlex, threading

class ProtocolAdapter:
    name = "base"
    def run_load(self, **kwargs):
        raise NotImplementedError

def run_subprocess(cmd, live_log=False):
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    def reader():
        for line in proc.stdout:
            out_lines.append(line)
            if live_log:
                print(line.rstrip())
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    return proc, out_lines
