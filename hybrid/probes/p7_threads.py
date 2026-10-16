"""P0 probe 7: how many threads does a BLAS matmul actually use?

Runs the measurement in child processes: the job's own environment, and a control
with OPENBLAS_NUM_THREADS=4 to show the probe can see extra threads."""
import json
import os
import subprocess
import sys
import threading
import time

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def threads_now():
    for line in open("/proc/self/status"):
        if line.startswith("Threads:"):
            return int(line.split()[1])


def measure(kind):
    peak = [0]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], threads_now())
            time.sleep(0.002)

    if kind == "numpy":
        import numpy as np
        a = np.random.default_rng(1).standard_normal((1200, 1200))
        configured = None
        work = lambda: a @ a
    else:
        import torch
        configured = torch.get_num_threads()
        a = torch.randn(1200, 1200, dtype=torch.float64)
        work = lambda: a @ a
    work()
    sampler = threading.Thread(target=sample)
    sampler.start()
    end = time.time() + 2.0
    products = 0
    while time.time() < end:
        work()
        products += 1
    stop.set()
    sampler.join()
    return {"kind": kind, "peak_threads_including_main_and_sampler": peak[0],
            "worker_threads": peak[0] - 2, "products": products, "configured_threads": configured}


if len(sys.argv) > 1:
    print(json.dumps(measure(sys.argv[1])))
    sys.exit(0)


def child(kind, env_override=None):
    env = dict(os.environ, **(env_override or {}))
    done = subprocess.run([sys.executable, __file__, kind], capture_output=True, text=True, env=env, check=True)
    return json.loads(done.stdout)


cgroup_cpu_max = open("/sys/fs/cgroup/cpu.max").read().strip()
result = {
    "env": {k: os.environ.get(k) for k in THREAD_VARS},
    "affinity_cpus": len(os.sched_getaffinity(0)),
    "cgroup_cpu_max": cgroup_cpu_max,
    "numpy": child("numpy"),
    "torch": child("torch"),
    "control_numpy_four_threads": child("numpy", {"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4"}),
}
json.dump(result, open("/out/result.json", "w"), sort_keys=True)
