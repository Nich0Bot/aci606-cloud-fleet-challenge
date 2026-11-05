from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time


@dataclass
class RobotState:
    robot_id: str
    x: float
    y: float
    heading: float
    battery: float
    health: float


def make_robot(index: int, rng: random.Random) -> RobotState:
    return RobotState(
        robot_id=f"R{index:02d}",
        x=rng.uniform(0, 10),
        y=rng.uniform(0, 10),
        heading=rng.uniform(-math.pi, math.pi),
        battery=rng.uniform(70, 100),
        health=rng.uniform(0.15, 0.95),
    )


def status_from_health(health: float) -> str:
    if health < 0.38:
        return "CRITICAL"
    if health < 0.62:
        return "WARNING"
    return "NORMAL"


def step_robot(robot: RobotState, rng: random.Random, dt: float = 1.0) -> tuple[dict, str]:
    robot.health += rng.gauss(0, 0.012)
    if rng.random() < 0.02:
        robot.health -= rng.uniform(0.05, 0.18)
    robot.health = max(0.05, min(0.98, robot.health))

    speed = max(0.05, min(1.25, rng.gauss(0.72 + 0.18 * robot.health, 0.12)))
    robot.heading += rng.gauss(0, 0.11)
    robot.x += math.cos(robot.heading) * speed * dt
    robot.y += math.sin(robot.heading) * speed * dt
    robot.x %= 10.0
    robot.y %= 10.0

    battery_drain = 0.035 + 0.025 * speed + (1 - robot.health) * 0.02
    robot.battery = max(0, robot.battery - battery_drain)

    motor_temp = 34 + 38 * (1 - robot.health) + 10 * speed + rng.gauss(0, 2.0)
    vibration = max(0, 0.12 + 1.65 * (1 - robot.health) + rng.gauss(0, 0.09))
    motor_current = max(0.2, 1.5 + 2.8 * speed + 2.2 * (1 - robot.health) + rng.gauss(0, 0.25))
    obstacle_distance = max(0.05, min(4.0, rng.expovariate(0.72)))

    sample = {
        "robot_id": robot.robot_id,
        "timestamp": time.time(),
        "x": round(robot.x, 4),
        "y": round(robot.y, 4),
        "speed": round(speed, 4),
        "battery": round(robot.battery, 3),
        "motor_temp": round(motor_temp, 3),
        "vibration": round(vibration, 4),
        "motor_current": round(motor_current, 4),
        "obstacle_distance": round(obstacle_distance, 4),
    }
    return sample, status_from_health(robot.health)
