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
"""In-memory record of recent grades — feeds the built-in dashboard.

A bounded ring buffer plus running counters, guarded by one lock. Deliberately
memory-only: the service stays stateless (restart = hot update = clean slate),
and the dashboard is an OBSERVATION window, not an archive — the durable record
of every grade lives on the training side (instance.log / the reward-shim
generation dump).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections import deque

MAX_RECORDS = int(os.getenv("DASHBOARD_MAX_RECORDS", "300"))  # ~300 × (shot ~200KB + raw ≤200KB) ≈ 120MB
MAX_RAW_CHARS = 200_000  # a 60k-token generation fits; anything longer is truncated mid-notice


class RecentGrades:
    def __init__(self, maxlen: int = MAX_RECORDS):
        self._lock = threading.Lock()
        self._items: deque[dict] = deque(maxlen=maxlen)
        self._stats = {
            "total": 0,
            "ok": 0,
            "drop": 0,
            "no_delivery": 0,
            "unrenderable": 0,
            "zero": 0,
            "pass": 0,
            "broken": 0,
            "rejected": 0,
            "client_gone": 0,
        }
        self._started = time.time()

    def add(self, record: dict, shot_jpg_b64: str | None, raw: str | None = None) -> None:
        """record: the /grade response fields + query.
        The shot and the raw generation ride along but are stripped from list
        views. Every record gets a unique ``_id`` — task_id is NOT unique (the
        n rollouts of one prompt share it), so payloads are served per record."""
        if raw and len(raw) > MAX_RAW_CHARS:
            raw = raw[:MAX_RAW_CHARS] + f"\n\n[... truncated, {len(raw)} chars total]"
        with self._lock:
            self._items.appendleft(
                {
                    **record,
                    "_id": uuid.uuid4().hex[:12],
                    "_shot_b64": shot_jpg_b64,
                    "_raw": raw,
                    "ts": round(time.time(), 3),
                }
            )
            s = self._stats
            s["total"] += 1
            # Classified on query_score, NOT on reward: in group mode `reward` is
            # always None here (it is the rank the driver computes once the
            # siblings land), so a `reward > 0` test would file every graded
            # rollout under "zero" and peg pass at 0.
            if record.get("status") != "ok":
                s["drop"] += 1
            elif record.get("no_delivery"):
                s["no_delivery"] += 1
            elif (record.get("signals") or {}).get("unrenderable"):
                s["unrenderable"] += 1
            elif ((record.get("runtime") or {}).get("factor")) == 0:
                # Rendered and judged, but the page's JS is dead — the gate zeroes
                # it downstream, so it is not a pass however well query scored.
                s["broken"] += 1
            elif (record.get("query_score") or 0) > 0:
                s["pass"] += 1
            else:
                s["zero"] += 1
            s["ok"] = s["total"] - s["drop"]

    def note_rejected(self) -> int:
        """Count a capacity 503 and return the running total.

        A rejected request never reaches grade_task, so it cannot ride along in
        add() — without this the service reports a healthy ~0.4% drop while the
        training side eats 503s for 40% of its samples (measured at
        judges_s ~130s at 3.3 req/s against HTTP_CONCURRENCY=256).
        """
        with self._lock:
            self._stats["rejected"] += 1
            return self._stats["rejected"]

    def note_client_gone(self) -> int:
        """Count a verdict we finished but could not deliver, and return the total.

        The grade succeeded and is in HISTORY; the peer had already closed the
        socket. `rejected` counts requests we never started — this one counts
        requests we completed at full cost and threw away, which is the more
        expensive of the two: it also means the client is retrying, so each
        occurrence is roughly a second full grade queued behind it.

        Measured shape: judges over the 180s x 3 budget push
        total_s past the client's 600s, the client hangs up and retries, and the
        retry lands on a pool that just demonstrated it cannot finish in time.
        """
        with self._lock:
            self._stats["client_gone"] += 1
            return self._stats["client_gone"]

    def recent(self, limit: int = 100) -> dict:
        """Newest-first metadata (no image/text payloads) + running stats."""
        with self._lock:
            items = [
                {k: v for k, v in it.items() if k not in ("_shot_b64", "_raw")}
                | {"has_shot": bool(it.get("_shot_b64")), "has_raw": bool(it.get("_raw"))}
                for it in list(self._items)[:limit]
            ]
            return {
                "stats": dict(self._stats, uptime_s=round(time.time() - self._started)),
                "items": items,
            }

    def _find(self, record_id: str) -> dict | None:
        for it in self._items:
            if it.get("_id") == record_id:
                return it
        return None

    def shot(self, record_id: str) -> bytes | None:
        import base64

        with self._lock:
            it = self._find(record_id)
            return base64.b64decode(it["_shot_b64"]) if it and it.get("_shot_b64") else None

    def raw(self, record_id: str) -> str | None:
        with self._lock:
            it = self._find(record_id)
            return it.get("_raw") if it else None
