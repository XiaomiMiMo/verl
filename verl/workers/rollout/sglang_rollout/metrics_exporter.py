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
"""Node-local Prometheus exporter bound to a fixed, platform-scraped metric port."""

from __future__ import annotations

import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess


class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return

        registry = CollectorRegistry()
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            multiprocess.MultiProcessCollector(registry)
        payload = generate_latest(registry)
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        return


def start_metrics_server(port: int) -> HTTPServer:
    """Start one exporter per Pod; duplicate TP actors reuse the existing port."""
    server = HTTPServer(("0.0.0.0", port), _MetricsHandler)
    Thread(target=server.serve_forever, name=f"verl-metrics-{port}", daemon=True).start()
    return server
