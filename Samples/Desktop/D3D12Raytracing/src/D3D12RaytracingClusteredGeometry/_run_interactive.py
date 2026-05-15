"""
Launch the sample interactively for visual inspection, with an auto-kill timeout
so the agent can't accidentally leave the EXE running forever (and so subsequent
build attempts don't fail with a locked-EXE error).

Usage:
  python _run_interactive.py [seconds]   # default 25s
"""
import subprocess
import sys
import time
import pathlib

DEFAULT_SECONDS = 25
exe = r"d:\samples\Samples\Desktop\D3D12Raytracing\src\D3D12RaytracingClusteredGeometry\bin\x64\Debug\D3D12RaytracingClusteredGeometry.exe"

duration = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SECONDS

print(f"[interactive] launching {exe}")
print(f"[interactive] will auto-kill after {duration:.0f}s")

p = subprocess.Popen([exe], cwd=str(pathlib.Path(exe).parent))
start = time.time()
try:
    rc = p.wait(timeout=duration)
    print(f"[interactive] exited cleanly with rc={rc} after {time.time()-start:.1f}s")
except subprocess.TimeoutExpired:
    elapsed = time.time() - start
    print(f"[interactive] timeout after {elapsed:.1f}s, terminating")
    p.terminate()
    try:
        p.wait(timeout=3)
        print("[interactive] terminated cleanly")
    except subprocess.TimeoutExpired:
        print("[interactive] terminate did not respond, killing")
        p.kill()
        p.wait()
        print("[interactive] killed")
