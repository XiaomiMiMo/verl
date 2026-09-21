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
"""Remote-asset localization for rendering. Carried over from a batch pipeline
where it survived a 750k-page run, so the parts worth keeping are known:

  * content-addressed shared cache by NORMALIZED url (renditions of one Unsplash
    photo collapse to one entry — measured 58% fewer fetches), two prefix-dir
    levels, atomic tmp+rename writes, `.meta` sidecars so a 404 is cached too;
  * DNS watchdog: hallucinated asset domains hang getaddrinfo OUTSIDE urlopen's
    timeout — resolve each host once behind a deadline and remember the verdict;
  * CSS deps resolved before the CSS is published, so fonts inside a stylesheet
    are rewritten exactly once.

Dropped: the multi-shard fetch gate (cluster politeness at 2000 workers; this
service runs a handful of renders at a time).

Why localize at all: a direct Unsplash fetch takes ~8s per image at render
time; a timeout shows the judge an empty frame it then blames on the page
(measured 3x inflated reject rate). Cache root: DESIGN_GRADER_ASSETS env.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ASSETS = os.environ.get("DESIGN_GRADER_ASSETS", "/tmp/design_grader_assets")

URL_RE = re.compile(r"""(https?://[^\s"'()<>\\]+)""")
IMPORTMAP_RE = re.compile(r"<script[^>]*type=[\"']importmap[\"'][^>]*>.*?</script>", re.S | re.I)
CSS_URL = re.compile(r"""url\(\s*['\"]?([^'\")]+)['\"]?\s*\)""")
ASSET_EXT = re.compile(r"\.(jpe?g|png|webp|avif|gif|svg|ico|bmp|woff2?|ttf|otf|mp4|webm)(\?|#|$)", re.I)
ASSET_HOST = re.compile(
    r"(images\.unsplash\.com|images\.pexels\.com|picsum\.photos|randomuser\.me|"
    r"i\.pravatar\.cc|placehold\.co|via\.placeholder\.com|dummyimage\.com|"
    r"ui-avatars\.com|source\.unsplash\.com|loremflickr\.com|robohash\.org)",
    re.I,
)
CODE_EXT = re.compile(r"\.(css|m?js|cjs)(\?|#|$)", re.I)
CODE_HOST = re.compile(
    r"(fonts\.googleapis\.com|fonts\.gstatic\.com|cdn\.jsdelivr\.net|"
    r"cdnjs\.cloudflare\.com|unpkg\.com|esm\.sh|cdn\.skypack\.dev|"
    r"cdn\.tailwindcss\.com)",
    re.I,
)
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
EXT_BY_CTYPE = {
    "text/css": ".css",
    "text/javascript": ".js",
    "application/javascript": ".js",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/avif": ".avif",
    "image/svg+xml": ".svg",
    "image/x-icon": ".ico",
    "font/woff2": ".woff2",
    "font/woff": ".woff",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}
UNSPLASH_Q = "?auto=format&fit=crop&w=1600&q=80"
MAX_ASSET_BYTES = 32 << 20


def wanted(u: str) -> bool:
    return bool(ASSET_EXT.search(u) or ASSET_HOST.search(u) or CODE_EXT.search(u) or CODE_HOST.search(u))


def normalize(u: str) -> tuple[str, str]:
    """(cache key, fetch url): collapse renditions of one asset onto one entry."""
    try:
        x = urllib.parse.urlparse(u)
    except Exception:
        return u, u
    host = (x.hostname or "").lower()
    if host == "images.unsplash.com":
        return "unsplash:" + x.path, "https://images.unsplash.com" + x.path + UNSPLASH_Q
    if host in (
        "picsum.photos",
        "i.pravatar.cc",
        "randomuser.me",
        "placehold.co",
        "via.placeholder.com",
        "ui-avatars.com",
        "dummyimage.com",
        "robohash.org",
        "loremflickr.com",
    ):
        return host + x.path, f"{x.scheme}://{host}{x.path}"  # generated: any one is fine
    if host in ("fonts.googleapis.com",):
        return u, u  # the family list IS the identity
    return f"{x.scheme}://{host}{x.path}", u  # drop cache-busting query


def paths_for(key: str) -> tuple[str, str]:
    h = hashlib.sha1(key.encode()).hexdigest()
    return os.path.join(ASSETS, h[:2], h[2:4]), h


def read_meta(key: str) -> dict | None:
    d, h = paths_for(key)
    try:
        with open(os.path.join(d, h + ".meta")) as f:
            return json.load(f)
    except Exception:
        return None


def write_atomic(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


_DNS: dict[str, bool] = {}
_dns_lock = threading.Lock()
_dns_pool = None

LOCALIZE_CONCURRENCY = 24
_FETCH_POOL = ThreadPoolExecutor(max_workers=LOCALIZE_CONCURRENCY, thread_name_prefix="fetch")


def host_resolves(url: str, wait: float = 6) -> bool:
    import socket
    from concurrent.futures import ThreadPoolExecutor as _TPE

    global _dns_pool
    try:
        host = urllib.parse.urlparse(url).hostname
    except Exception:
        return False
    if not host:
        return False
    with _dns_lock:
        if host in _DNS:
            return _DNS[host]
        if _dns_pool is None:
            _dns_pool = _TPE(8)
    fut = _dns_pool.submit(socket.getaddrinfo, host, 443)
    try:
        fut.result(timeout=wait)
        ok = True
    except Exception:
        ok = False  # NXDOMAIN and too-slow both mean: don't let this host hold a thread
    with _dns_lock:
        _DNS[host] = ok
    return ok


def http_get(url: str, timeout: float = 45):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        chunks, total = [], 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            total += len(b)
            if total > MAX_ASSET_BYTES:
                raise ValueError(f"asset exceeds {MAX_ASSET_BYTES >> 20}MB cap")
            chunks.append(b)
        return b"".join(chunks), (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()


def fetch_one(raw_url: str, depth: int = 0, deadline: float | None = None) -> dict:
    """Cache-or-fetch. Returns the meta dict (`rel` set when the bytes are on disk).

    `deadline` is an absolute unix time by which every network touch must be
    done. Past it we return the miss instead of fetching: a page that renders
    with a few broken images still gets judged, a page that never renders is a
    drop. One cold page used to cost N x 45s serially (a CSS pulls its fonts
    recursively), which held an HTTP slot long enough for the client to give up
    and for the whole service to back up behind it.
    """
    key, fetch_url = normalize(raw_url)
    have = read_meta(key)
    if have is not None:
        return have
    if deadline is not None and time.time() >= deadline:
        return {"key": key, "url": fetch_url, "miss": "localize_budget"}
    d, h = paths_for(key)
    meta: dict = {"key": key, "url": fetch_url}
    try:
        if not host_resolves(fetch_url):
            raise ValueError("dns_unresolvable")
        budget_left = (deadline - time.time()) if deadline is not None else 45.0
        if budget_left <= 0:
            return {"key": key, "url": fetch_url, "miss": "localize_budget"}
        data, ctype = http_get(fetch_url, timeout=min(45.0, budget_left))
        ext = EXT_BY_CTYPE.get(ctype) or (os.path.splitext(urllib.parse.urlparse(fetch_url).path)[1][:6] or ".bin")
        fname = h + ext
        if ext == ".css" and depth == 0:
            body = data.decode("utf-8", "replace")
            deps = {}
            for ref in set(CSS_URL.findall(body)):
                if ref.startswith("data:"):
                    continue
                dep = fetch_one(urllib.parse.urljoin(fetch_url, ref), depth + 1, deadline=deadline)
                if dep.get("rel"):
                    deps[ref] = "../../" + dep["rel"]

            def sub(m):
                r = m.group(1)
                return f"url({deps[r]})" if r in deps else m.group(0)

            data = CSS_URL.sub(sub, body).encode()
        write_atomic(os.path.join(d, fname), data)
        meta.update(status=200, bytes=len(data), ctype=ctype, rel=os.path.join(h[:2], h[2:4], fname))
    except Exception as e:  # noqa: BLE001 — every failure is a cachable verdict
        meta.update(status=getattr(e, "code", None) or 0, rel=None, error=f"{type(e).__name__}: {e}"[:120])
    try:
        write_atomic(os.path.join(d, h + ".meta"), json.dumps(meta).encode())
    except Exception:
        pass
    return meta


def localize(dist_dir: str) -> dict:
    """Fetch every wanted remote url referenced by index.html, then rewrite the
    html to the on-disk copies (via a `_assets` symlink into the shared cache).
    importmap blocks are left untouched (rewriting them breaks module graphs).

    Fetches go through the process-wide _FETCH_POOL (LOCALIZE_CONCURRENCY, 12)
    and the whole pass is bounded by LOCALIZE_BUDGET_S (default 20s). Serially
    this used to cost N x ~1s per cold page, which routinely exhausted the
    budget: the skipped urls then render as broken images and the judge scores
    the page for damage the model never caused. Parallel, 20 cold urls land in
    ~2s, well inside the budget.

    Past the deadline the remaining urls stay remote — a page with a few broken
    images is still gradeable, an unrendered page is not. Because the pool is
    shared, thread count stays flat no matter how many requests localize at once.

    The budget caps NEW fetches, not in-flight ones: with more urls than pool
    slots the overflow queues, and each queued worker re-checks the deadline
    before it touches the network, so the tail is dropped rather than dragged
    out. Measured: 30 urls against a 12-slot pool with a 1s budget starts 12,
    skips 18, returns in 1.01s.
    """
    budget = float(os.getenv("LOCALIZE_BUDGET_S", "20"))
    deadline = time.time() + budget
    entry = os.path.join(dist_dir, "index.html")
    html = open(entry, errors="replace").read()
    hit: dict[str, str] = {}
    targets = [u.rstrip(".,;") for u in set(URL_RE.findall(html)) if wanted(u.rstrip(".,;"))]
    n_skipped = 0
    if targets:
        futures = {_FETCH_POOL.submit(fetch_one, u, 0, deadline): u for u in targets}
        for fut, u in futures.items():
            try:
                meta = fut.result(timeout=max(0.0, deadline - time.time()) + 5)
            except Exception:  # noqa: BLE001 — one bad url must not sink the page
                n_skipped += 1
                continue
            if meta.get("rel"):
                hit[u] = meta["rel"]
            elif meta.get("miss") == "localize_budget":
                n_skipped += 1
    n_hit = n_miss = 0

    def sub(m):
        nonlocal n_hit, n_miss
        u = m.group(1)
        clean = u.rstrip(".,;")
        if not wanted(clean):
            return m.group(0)
        rel = hit.get(clean)
        if rel:
            n_hit += 1
            return "./_assets/" + rel + u[len(clean) :]
        n_miss += 1
        return m.group(0)

    out, pos = [], 0
    for m in IMPORTMAP_RE.finditer(html):
        out.append(URL_RE.sub(sub, html[pos : m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(URL_RE.sub(sub, html[pos:]))
    with open(entry, "w") as f:
        f.write("".join(out))
    link = os.path.join(dist_dir, "_assets")
    if not os.path.islink(link) and not os.path.exists(link):
        os.symlink(os.path.abspath(ASSETS), link)
    return {"localized": n_hit, "left_remote": n_miss, **({"localize_skipped": n_skipped} if n_skipped else {})}
