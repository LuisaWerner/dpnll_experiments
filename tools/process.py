import os
import signal
import subprocess
import sys
import psutil

active_subprocess = {}


def terminate(proc):
    id_proc = proc.pid
    try:
        if os.name == 'nt':
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        del active_subprocess[id_proc]
    except Exception as e:
        print(f"Error terminating process {proc.pid}: {e}")


def handle_termination(signum, frame):
    print("Received termination signal. Cleaning up...")
    for proc in active_subprocess.values():
        terminate(proc)
    sys.exit(0)


signal.signal(signal.SIGINT, handle_termination)
signal.signal(signal.SIGTERM, handle_termination)


# Example of starting a subprocess
def start_subprocess(cmd: list[str]):
    if os.name == 'nt':
        proc = subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    else:
        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    active_subprocess[proc.pid] = proc
    return proc