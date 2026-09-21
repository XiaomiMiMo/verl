#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Probe an already-running Ray cluster before launching a multi-node job.

    python3 cluster_precheck.py --nnodes 8 --gpus-per-node 8 \
        --model /path/to/policy-checkpoint

Prints a human report to stderr and exactly one line to **stdout**:

    LD_LIBRARY_PATH=<value>      # the value Ray workers need, or empty if they need none

so a launcher can do  `LD=$(python3 cluster_precheck.py ...)`  and forward it verbatim.
Exits non-zero (with the reason) if the cluster cannot run the job.

Why each check exists -- all four were real, silent failures on this cluster:

1. **`ray.init()` does not find an out-of-band cluster.** With `ray start --head` already
   running in this pod, a bare `ray.init()` still spun up a *second*, 1-node, 0-GPU local
   instance. verl then reported 0 GPUs available and the placement group hung forever with no
   error. So the launcher must pass `address` explicitly; this script asserts that
   `address="auto"` really lands on the big cluster.

2. **CUDA forward-compat does not reach Ray workers.** The hosts run driver 570.x (CUDA 12.8)
   while the image's torch is 2.11+cu130 (needs r580). `/opt/cuda_compat.sh` fixes this by
   prepending /usr/local/cuda/compat to LD_LIBRARY_PATH, but Ray workers started
   out-of-band inherit the *raylet's* environment, not the launcher shell's -- measured
   `torch.cuda.is_available() == False` on all 8 workers. LD_LIBRARY_PATH is read by the
   dynamic loader at exec time, so it cannot be fixed from inside Python; it has to be in
   `runtime_env.env_vars`. This script derives the value by *running the image's own
   cuda_compat.sh on a worker* (rather than hardcoding a path), then verifies torch actually
   sees the GPU with it.

3. **The head node has no GPU.** verl decides whether to set CUDA_DEVICE_MAX_CONNECTIONS=1
   from the *driver's* compute capability (`constants_ppo.py`, `_is_hopper_or_ampere`). On a
   GPU-less head that reads (None, None), so the Megatron workers -- which are Hopper and do
   need it -- never get it. The launcher forwards it unconditionally; this script reports the
   worker capability so the claim is checked against reality, not assumed.

4. **Shared-storage visibility is per-node.** The model, the parquet, the kubeconfig and the
   dump dir all live on a shared network filesystem. A node that mounted it late looks fine to `ls` from the head
   and fails 20 minutes into the run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

CUDA_COMPAT_SH = "/opt/cuda_compat.sh"


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def _probe_payload(paths: list[str], extra_modules: list[str], pythonpath: list[str]) -> dict:
    """Runs *on a worker*, holding one GPU. Never raises; reports instead."""
    out: dict = {"host": os.uname().nodename, "pid": os.getpid()}

    for entry in reversed([p for p in pythonpath if p]):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)

    ld_before = os.environ.get("LD_LIBRARY_PATH", "")
    ld_after = ld_before
    if os.path.exists(CUDA_COMPAT_SH):
        try:
            ld_after = subprocess.run(
                ["bash", "-c", f'. {CUDA_COMPAT_SH} >/dev/null 2>&1; printf "%s" "${{LD_LIBRARY_PATH:-}}"'],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            ).stdout
        except Exception as e:  # noqa: BLE001
            out["compat_error"] = repr(e)
    else:
        out["compat_error"] = f"{CUDA_COMPAT_SH} not found in the image"
    out["ld_before"] = ld_before
    out["ld_after"] = ld_after
    out["ld_changed"] = ld_after != ld_before

    try:
        import torch

        out["cuda_now"] = bool(torch.cuda.is_available())
    except Exception as e:  # noqa: BLE001
        out["cuda_now"] = False
        out["torch_error"] = repr(e)

    env = dict(os.environ, LD_LIBRARY_PATH=ld_after)
    code = (
        "import torch,json;"
        "ok=torch.cuda.is_available();"
        "p=torch.cuda.get_device_properties(0) if ok else None;"
        "print(json.dumps({'ok':ok,'name':getattr(p,'name',None),"
        "'gib':round(p.total_memory/2**30,1) if p else None,"
        "'cap':list(torch.cuda.get_device_capability(0)) if ok else None,"
        "'count':torch.cuda.device_count() if ok else 0}))"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)
        import json as _json

        out["with_fix"] = _json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        out["with_fix"] = {"ok": False}
        out["with_fix_error"] = f"{e!r} stderr={r.stderr[-400:] if 'r' in dir() else ''}"

    out["paths"] = {p: os.path.exists(p) for p in paths}
    modules = ["verl", "megatron.core", "megatron.bridge", "fla", "sglang", "mimoagent", *extra_modules]
    for mod in dict.fromkeys(modules):
        try:
            __import__(mod)
            out.setdefault("imports", {})[mod] = True
        except Exception as e:  # noqa: BLE001
            out.setdefault("imports", {})[mod] = f"ERR {type(e).__name__}: {e}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", default="auto")
    ap.add_argument("--nnodes", type=int, required=True)
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--model", default=None, help="checkpoint dir that every node must be able to see")
    ap.add_argument("--path", action="append", default=[], help="extra path every node must see (repeatable)")
    ap.add_argument("--module", action="append", default=[], help="extra Python module every node must import")
    ap.add_argument(
        "--pythonpath",
        action="append",
        default=[],
        help="source path to add while probing imports (repeatable; pass the launcher PYTHONPATH entries)",
    )
    args = ap.parse_args()

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address=args.address, ignore_reinit_error=True, log_to_driver=False)

    gpu_nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU")]
    total_gpu = int(sum(n["Resources"]["GPU"] for n in gpu_nodes))
    want = args.nnodes * args.gpus_per_node
    log(f"[cluster] address={args.address}  GPU nodes={len(gpu_nodes)}  total GPUs={total_gpu}  (need {want})")
    for n in gpu_nodes:
        r = n["Resources"]
        log(f"  {n['NodeManagerAddress']:>16}  GPU={int(r['GPU'])}  CPU={int(r.get('CPU', 0))}")
    if total_gpu < want:
        return _die(f"cluster has {total_gpu} GPUs, the job asks for {args.nnodes}x{args.gpus_per_node}={want}")
    off = [n["NodeManagerAddress"] for n in gpu_nodes if int(n["Resources"]["GPU"]) != args.gpus_per_node]
    if off:
        log(f"[cluster] WARNING: these nodes do not have exactly {args.gpus_per_node} GPUs: {off}")
        log("          verl allocates one placement-group bundle per node; uneven nodes will not pack.")

    paths = [p for p in ([args.model] if args.model else []) + list(args.path) if p]
    probe = ray.remote(num_cpus=1, num_gpus=1)(_probe_payload)
    results = ray.get(
        [
            probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)).remote(
                paths, args.module, args.pythonpath
            )
            for n in gpu_nodes
        ]
    )

    ld_values, bad = set(), []
    for n, res in zip(gpu_nodes, results, strict=True):
        addr = n["NodeManagerAddress"]
        fix = res.get("with_fix", {})
        tag = "OK " if fix.get("ok") else "BAD"
        log(
            f"[{tag}] {addr:>16} {res['host'].split('.')[0]:<28} "
            f"cuda_now={res['cuda_now']}  with_fix={fix.get('ok')}  "
            f"{fix.get('name')} {fix.get('gib')}GiB cap={fix.get('cap')}"
        )
        if res.get("compat_error"):
            log(f"      compat: {res['compat_error']}")
        if not fix.get("ok"):
            bad.append(f"{addr}: torch.cuda unavailable even with LD_LIBRARY_PATH={res['ld_after']!r}")
        elif fix.get("count") != args.gpus_per_node and int(n["Resources"]["GPU"]) == args.gpus_per_node:
            pass
        ld_values.add(res["ld_after"])
        missing = [p for p, ok in res["paths"].items() if not ok]
        if missing:
            bad.append(f"{addr}: cannot see {missing}")
        broken = {m: v for m, v in (res.get("imports") or {}).items() if v is not True}
        if broken:
            bad.append(f"{addr}: import failures {broken}")

    if bad:
        return _die("\n  ".join(["worker probes failed:"] + bad))

    if len(ld_values) != 1:
        return _die(f"nodes disagree on the required LD_LIBRARY_PATH, cannot forward one value: {ld_values}")
    ld = ld_values.pop()

    caps = {tuple(r["with_fix"]["cap"]) for r in results}
    log(f"[cluster] compute capability on workers: {caps}")
    if caps and all(c[0] in (8, 9) for c in caps):
        log("[cluster] Hopper/Ampere -> workers DO need CUDA_DEVICE_MAX_CONNECTIONS=1;")
        log("          the driver on a GPU-less head cannot detect this, so the launcher forwards it.")

    log("[cluster] all checks passed")
    print(ld)  # the one stdout line
    return 0


def _die(msg: str) -> int:
    log("")
    log(f"[cluster] FAIL: {msg}")
    log("")
    print("")  # keep stdout's contract: one line, empty
    return 1


if __name__ == "__main__":
    sys.exit(main())
