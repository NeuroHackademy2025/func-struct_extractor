import os
import subprocess

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def run_bash(cmd):
    ret = subprocess.run(cmd, check=True)
    return ret.returncode
