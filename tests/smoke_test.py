from __future__ import annotations

import argparse
import time

import requests

SAMPLE = {
    "robot_id": "TEST-01",
    "timestamp": time.time(),
    "x": 1.2,
    "y": 3.4,
    "speed": 0.72,
    "battery": 81.0,
    "motor_temp": 49.0,
    "vibration": 0.42,
    "motor_current": 3.8,
    "obstacle_distance": 1.3,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://127.0.0.1:8000")
    args = p.parse_args()
    base = args.api.rstrip("/")

    checks = [
        ("GET /health", requests.get(base + "/health", timeout=5)),
        ("POST /telemetry", requests.post(base + "/telemetry", json=SAMPLE, timeout=5)),
        ("GET /robots", requests.get(base + "/robots", timeout=5)),
        ("GET /metrics", requests.get(base + "/metrics", timeout=5)),
    ]

    failed = False
    for name, response in checks:
        print(f"{name:<20} -> {response.status_code}")
        if not response.ok:
            failed = True
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
