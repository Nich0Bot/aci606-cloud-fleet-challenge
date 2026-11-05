from __future__ import annotations

import argparse
import random
import time

import requests

from core import make_robot, step_robot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Virtual robot fleet telemetry generator")
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--robots", type=int, default=5)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=0, help="0 = run until Ctrl+C")
    p.add_argument("--seed", type=int, default=606)
    p.add_argument("--timeout", type=float, default=3.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    robots = [make_robot(i + 1, rng) for i in range(args.robots)]
    endpoint = args.api.rstrip("/") + "/telemetry"

    print(f"Sending telemetry from {len(robots)} robots to {endpoint}")
    print("Ctrl+C to stop.\n")

    step = 0
    try:
        while args.steps <= 0 or step < args.steps:
            step += 1
            ok = 0
            for robot in robots:
                sample, _ = step_robot(robot, rng, args.interval)
                try:
                    response = requests.post(endpoint, json=sample, timeout=args.timeout)
                    if response.ok:
                        ok += 1
                    else:
                        print(f"{robot.robot_id}: HTTP {response.status_code} {response.text[:100]}")
                except requests.RequestException as exc:
                    print(f"{robot.robot_id}: connection error: {exc}")
            print(f"step={step:04d} sent={ok}/{len(robots)}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nSimulator stopped.")


if __name__ == "__main__":
    main()
