"""
hello_worker.py
===============
A minimal forward-run script that verifies the conda-pack environment
was unpacked and activated correctly on the HTCondor worker slot.

This is the stand-in for forward_run.py in a real pestpp run.
pestpp-ies calls it once per realisation on each worker.

It checks:
  1. Python version and interpreter path (confirms env is active)
  2. All packages from WORKER_ENV_PACKAGES are importable
  3. Basic numpy / pandas / flopy / geopandas / pyemu operations
  4. Writes a result file so the master can see the worker ran cleanly

Exit 0 on success, non-zero on any failure (pestpp will mark the
realisation as failed and potentially freeze/retry).

Outputs
-------
hello_result.txt
    Human-readable log — written on the worker's local scratch directory.
    NOT transferred back to the master (only visible in worker dirs or
    HTCondor per-job stdout logs).
hello_obs.dat
    Three numeric lines read by PEST via hello_obs.ins.  This IS
    collected by the master via the normal PEST I/O mechanism.
"""
import os
import sys
import socket
import datetime

print(f"[hello_worker] host={socket.gethostname()} pid={os.getpid()}")
print(f"[hello_worker] python={sys.executable}  version={sys.version.split()[0]}")
print(f"[hello_worker] started={datetime.datetime.utcnow().isoformat()}")

# -----------------------------------------------------------------------
# 1. Check all expected packages are importable and print versions
# -----------------------------------------------------------------------
checks = {}
try:
    import numpy as np
    checks["numpy"] = np.__version__
except ImportError as e:
    checks["numpy"] = f"FAILED: {e}"

try:
    import pandas as pd
    checks["pandas"] = pd.__version__
except ImportError as e:
    checks["pandas"] = f"FAILED: {e}"

try:
    import matplotlib
    checks["matplotlib"] = matplotlib.__version__
except ImportError as e:
    checks["matplotlib"] = f"FAILED: {e}"

try:
    import flopy
    checks["flopy"] = flopy.__version__
except ImportError as e:
    checks["flopy"] = f"FAILED: {e}"

try:
    import geopandas as gpd
    checks["geopandas"] = gpd.__version__
except ImportError as e:
    checks["geopandas"] = f"FAILED: {e}"

try:
    import pyemu
    checks["pyemu"] = pyemu.__version__
except ImportError as e:
    checks["pyemu"] = f"FAILED: {e}"

print("\n[hello_worker] Package versions:")
failed = []
for pkg, ver in checks.items():
    status = "✓" if not ver.startswith("FAILED") else "✗"
    print(f"  {status} {pkg}: {ver}")
    if ver.startswith("FAILED"):
        failed.append(pkg)

# -----------------------------------------------------------------------
# 2. Simple numpy / pandas smoke test
# -----------------------------------------------------------------------
print("\n[hello_worker] Running smoke tests …")
try:
    arr = np.arange(10).reshape(2, 5)
    assert arr.sum() == 45
    df = pd.DataFrame({"a": np.random.rand(5), "b": np.random.rand(5)})
    assert len(df) == 5
    print("  ✓ numpy / pandas OK")
except Exception as e:
    print(f"  ✗ numpy/pandas smoke test FAILED: {e}")
    failed.append("numpy/pandas smoke")

# -----------------------------------------------------------------------
# 3. Write result file (pestpp reads forward model outputs from files)
# -----------------------------------------------------------------------
result = {
    "host": socket.gethostname(),
    "python": sys.executable,
    "packages": checks,
    "smoke_test": "PASS" if not failed else f"FAIL: {failed}",
    "finished": datetime.datetime.utcnow().isoformat(),
}

# Write human-readable result log
with open("hello_result.txt", "w") as f:
    for k, v in result.items():
        f.write(f"{k}: {v}\n")

# Write numeric observations file for PEST instruction file.
# Each line is a single float PEST can parse:
#   obs_n_packages  — number of packages that imported successfully
#   obs_n_failed    — number of failed package imports (0 = all good)
#   obs_smoke       — 1.0 = smoke tests passed, 0.0 = failed
n_ok = sum(1 for v in checks.values() if not v.startswith("FAILED"))
n_fail = len(checks) - n_ok
smoke_val = 0.0 if failed else 1.0
with open("hello_obs.dat", "w") as f:
    f.write(f"{float(n_ok)}\n")
    f.write(f"{float(n_fail)}\n")
    f.write(f"{smoke_val}\n")

print(f"\n[hello_worker] Result written to hello_result.txt and hello_obs.dat")

if failed:
    print(f"\n[hello_worker] FAILED checks: {failed}")
    sys.exit(1)

print("[hello_worker] All checks passed — exit 0")
sys.exit(0)

