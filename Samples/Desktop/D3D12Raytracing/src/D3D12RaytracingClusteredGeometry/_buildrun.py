
import subprocess, pathlib, os, sys, tempfile, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

msbuild = r"C:\Program Files\Microsoft Visual Studio\18\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
sln = r"d:\samples\Samples\Desktop\D3D12Raytracing\src\D3D12RaytracingClusteredGeometry\D3D12RaytracingClusteredGeometry.sln"
exe = r"d:\samples\Samples\Desktop\D3D12Raytracing\src\D3D12RaytracingClusteredGeometry\bin\x64\Debug\D3D12RaytracingClusteredGeometry.exe"
shots_dir = pathlib.Path(r"d:\samples\Samples\Desktop\D3D12Raytracing\src\D3D12RaytracingClusteredGeometry\screenshots")
shots_dir.mkdir(parents=True, exist_ok=True)
# Unique per-run filename to defeat any path-based caching in tools
shot = str(shots_dir / f"m2_{int(time.time())}.png")
log = pathlib.Path(tempfile.gettempdir()) / "D3D12RaytracingClusteredGeometry.log"

# Build
r = subprocess.run([msbuild, sln, "/p:Configuration=Debug", "/p:Platform=x64",
                    "/v:m", "/nologo", "/clp:Summary;ShowCommandLine=false;PerformanceSummary=false"],
                   capture_output=True, text=True, timeout=300)
print("BUILD rc=", r.returncode)
if r.returncode != 0:
    out = r.stdout
    i = out.find("Project Evaluation")
    if i > 0:
        out = out[:i]
    print(out[-3000:])
    raise SystemExit(1)
else:
    print("(build ok)")

# Cleanup outputs
if log.exists():
    os.remove(log)

# Run
r = subprocess.run([exe, "--screenshot", "5", shot], capture_output=True, text=True, timeout=30,
                   cwd=str(pathlib.Path(exe).parent))
print(f"\nRUN rc={r.returncode}")
print(f"shot exists: {pathlib.Path(shot).exists()} ({pathlib.Path(shot).stat().st_size if pathlib.Path(shot).exists() else 0} bytes)")
print(f"  -> {shot}")

# Dump log
print(f"\n--- log ({log}) ---")
if log.exists():
    raw = log.read_bytes()
    text = raw[2:].decode('utf-16-le', errors='replace') if raw[:2]==b'\xff\xfe' else raw.decode('utf-8', errors='replace')
    print(text)
else:
    print("(no log file)")
