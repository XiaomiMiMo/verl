#!/usr/bin/env bash
# Conditionally put CUDA's forward-compatibility libs on the loader path.
#
#   source /opt/cuda_compat.sh
#
# Why this exists: the base image's torch is built against a newer CUDA than some hosts'
# drivers provide. Measured on a 570.124.06 (CUDA 12.8) host with torch 2.11.0+cu130: every
# import succeeds, and `torch.cuda.is_available()` is quietly **False** --
#   "The NVIDIA driver on your system is too old (found version 12080)"
# The fix is already in the image (/usr/local/cuda/compat/libcuda.so.580.82.07, i.e. r580
# userspace), it is just not on LD_LIBRARY_PATH.
#
# Why it is CONDITIONAL rather than a plain `ENV LD_LIBRARY_PATH=...` in the Dockerfile:
# forward compatibility runs a NEW userspace libcuda on an OLD kernel driver. Once the host
# driver is newer than the compat lib, loading the compat lib is the wrong direction and can
# fail outright. Baking it unconditionally would trade today's broken hosts for tomorrow's.
# So: compare versions, and only prepend when the driver is actually behind.
#
# Safe to source repeatedly and on CPU-only hosts (nvidia-smi missing -> no-op).
#
# Deliberately contains **no pipelines into `head`/`sed 1q`**. This file is sourced by callers
# that run with `set -euo pipefail`; there, `nvidia-smi | head -1` makes head exit first,
# nvidia-smi dies of SIGPIPE, the pipeline reports 141, and `set -e` aborts the *caller* --
# which looks like the script produced no output at all, with no error anywhere. Take the first
# line with parameter expansion instead.

_cc_dir=/usr/local/cuda/compat
if [ -d "$_cc_dir" ] && command -v nvidia-smi >/dev/null 2>&1; then
    _cc_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)
    _cc_driver=${_cc_driver%%$'\n'*}            # first GPU's driver version
    _cc_driver_major=${_cc_driver%%.*}
    # e.g. libcuda.so.580.82.07 -> 580; take the highest if several are shipped.
    _cc_lib_major=
    for _cc_so in "$_cc_dir"/libcuda.so.*; do
        [ -e "$_cc_so" ] || continue           # unmatched glob
        _cc_v=${_cc_so##*/libcuda.so.}
        _cc_v=${_cc_v%%.*}
        case "$_cc_v" in
            ''|*[!0-9]*) continue ;;           # not a version-suffixed lib
        esac
        if [ -z "$_cc_lib_major" ] || [ "$_cc_v" -gt "$_cc_lib_major" ]; then
            _cc_lib_major=$_cc_v
        fi
    done

    case "${_cc_driver_major}:${_cc_lib_major}" in
        *:|:*)
            echo "[cuda_compat] could not read driver (${_cc_driver:-?}) or compat version; not touching LD_LIBRARY_PATH" >&2
            ;;
        *)
            if [ "$_cc_driver_major" -lt "$_cc_lib_major" ]; then
                case ":${LD_LIBRARY_PATH:-}:" in
                    *":$_cc_dir:"*) ;;  # already there
                    *) export LD_LIBRARY_PATH="${_cc_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
                esac
                echo "[cuda_compat] driver ${_cc_driver} < compat ${_cc_lib_major}: using forward-compat libs"
            else
                echo "[cuda_compat] driver ${_cc_driver} >= compat ${_cc_lib_major}: using the host driver"
            fi
            ;;
    esac
    unset _cc_driver _cc_driver_major _cc_lib_major _cc_so _cc_v
fi
unset _cc_dir
