#!/bin/bash
echo "=== CPU ==="; nproc; grep -E "^(cpu.max|cpu_quota)" /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null; cat /sys/fs/cgroup/cpu.max 2>/dev/null; cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null
echo "=== MEMORY limit (cgroup) ==="; cat /sys/fs/cgroup/memory.max 2>/dev/null; cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null
echo "=== MEMORY now ==="; free -g; cat /sys/fs/cgroup/memory.current 2>/dev/null; cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null
echo "=== OOM events (cgroup) ==="; cat /sys/fs/cgroup/memory.events 2>/dev/null; grep -E "oom|under_oom" /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null
echo "=== dmesg OOM (may need root) ==="; dmesg 2>/dev/null | grep -i -E "out of memory|oom-kill|killed process" | tail -5
echo "=== /tmp ==="; df -h /tmp; du -sh /tmp 2>/dev/null; ls /tmp | sed 's/[0-9a-z_]*$//' | sort | uniq -c | sort -rn | head -8
echo "=== leaked probe/playwright dirs ==="; ls -d /tmp/design_probe_* 2>/dev/null | wc -l; ls -d /tmp/design_grade_* 2>/dev/null | wc -l; ls -d /tmp/playwright* 2>/dev/null | wc -l; du -sh /tmp/playwright* 2>/dev/null | sort -h | tail -3
echo "=== processes ==="; ps -eo comm --no-headers | sort | uniq -c | sort -rn | head -8
echo "chrome RSS total MB:"; ps -eo rss,comm --no-headers | awk '/chrome/ {s+=$1} END {print s/1024}'
echo "=== grader ==="; curl -s -m 5 http://127.0.0.1:80/healthz 2>/dev/null || curl -s -m 5 http://127.0.0.1:8080/healthz 2>/dev/null; echo
