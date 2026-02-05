import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


@dataclass
class VehicleState:
    x: float
    y: float
    heading: float  # radians
    v: float


@dataclass
class ControlInput:
    steering: float  # radians
    accel: float  # m/s^2


class KinematicBicycle:
    def __init__(self, wheelbase: float) -> None:
        self.wheelbase = wheelbase

    def step(self, state: VehicleState, control: ControlInput, dt: float) -> VehicleState:
        v_next = state.v + control.accel * dt
        v_next = max(v_next, 0.0)
        beta = math.atan(0.5 * math.tan(control.steering))
        x_next = state.x + v_next * math.cos(state.heading + beta) * dt
        y_next = state.y + v_next * math.sin(state.heading + beta) * dt
        heading_next = state.heading + (v_next / self.wheelbase) * math.sin(beta) * dt
        return VehicleState(x_next, y_next, heading_next, v_next)


def generate_target_path(num_points: int = 500):
    xs = [i * 0.2 for i in range(num_points)]
    ys = [2.0 * math.sin(0.2 * x) for x in xs]
    headings = []
    for i in range(num_points):
        if i == 0:
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
        else:
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
        headings.append(math.atan2(dy, dx))
    return xs, ys, headings


def simple_controller(state: VehicleState, target_x: float, target_y: float, target_heading: float):
    dx = target_x - state.x
    dy = target_y - state.y
    target_angle = math.atan2(dy, dx)
    heading_error = (target_angle - state.heading + math.pi) % (2 * math.pi) - math.pi
    steering = max(min(heading_error, math.radians(30)), math.radians(-30))
    speed_target = 5.0
    accel = 1.0 * (speed_target - state.v)
    accel = max(min(accel, 2.0), -3.0)
    return ControlInput(steering, accel)


def run_simulation():
    dt = 0.1
    model = KinematicBicycle(wheelbase=2.6)
    target_xs, target_ys, target_headings = generate_target_path()

    state = VehicleState(x=0.0, y=-2.0, heading=0.0, v=0.0)
    states = [state]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(target_xs, target_ys, "--", color="gray", label="Target path")
    vehicle_point, = ax.plot([], [], "ro", label="Vehicle")
    heading_line, = ax.plot([], [], "r-", linewidth=1)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Kinematic Bicycle Tracking")
    ax.legend(loc="upper left")

    def update(frame_idx: int):
        nonlocal state
        target_x = target_xs[min(frame_idx, len(target_xs) - 1)]
        target_y = target_ys[min(frame_idx, len(target_ys) - 1)]
        target_heading = target_headings[min(frame_idx, len(target_headings) - 1)]

        control = simple_controller(state, target_x, target_y, target_heading)
        state = model.step(state, control, dt)
        states.append(state)

        vehicle_point.set_data([state.x], [state.y])
        heading_length = 1.0
        hx = state.x + heading_length * math.cos(state.heading)
        hy = state.y + heading_length * math.sin(state.heading)
        heading_line.set_data([state.x, hx], [state.y, hy])

        ax.set_xlim(state.x - 10, state.x + 10)
        ax.set_ylim(state.y - 10, state.y + 10)
        return vehicle_point, heading_line

    FuncAnimation(fig, update, frames=len(target_xs), interval=50, blit=True)
    plt.show()


if __name__ == "__main__":
    run_simulation()
