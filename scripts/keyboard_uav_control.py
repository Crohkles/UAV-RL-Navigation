#!/usr/bin/env python3
"""Basic keyboard teleop for AirSim multirotor.

This script is independent from RL training and provides game-like manual control.
It reads keyboard input in a non-blocking terminal loop and sends velocity/yaw-rate
commands to a drone.

Key map (game style):
- W/S: forward/backward (x)
- A/D: left/right (y)
- R/F: up/down (z in NED, up is negative z velocity)
- Q/E: yaw left/right (rate)
- Arrow keys: also supported for planar movement
- Space: hover (reset velocity and yaw-rate command)
- X: hard stop command (same as hover)
- T: takeoff
- L: land
- +/-: decrease/increase max speed
- H: print help
- Esc: quit
"""

from __future__ import annotations

import argparse
import curses
import math
import time
from collections import deque
from dataclasses import dataclass

import airsim


@dataclass
class CommandState:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate_deg: float = 0.0


class KeyboardTeleop:
    def __init__(
        self,
        vehicle_name: str,
        max_speed: float,
        max_yaw_rate_deg: float,
        dt: float,
        auto_takeoff: bool,
    ) -> None:
        self.vehicle_name = vehicle_name
        self.max_speed = max_speed
        self.max_yaw_rate_deg = max_yaw_rate_deg
        self.dt = dt
        self.auto_takeoff = auto_takeoff

        self.client = airsim.MultirotorClient()
        self.cmd = CommandState()
        self.running = True
        self.prev_collision = False
        self._recent_events: deque[str] = deque(maxlen=6)

        # Smoothing factor for command ramp-up/ramp-down.
        self.alpha = 0.35

    def connect(self) -> None:
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

        if self.auto_takeoff:
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()

    def shutdown(self) -> None:
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        except Exception:
            pass
        try:
            self.client.armDisarm(False, vehicle_name=self.vehicle_name)
            self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
        except Exception:
            pass

    def print_help(self) -> None:
        self._add_event("KEYMAP: W/S A/D R/F Q/E + Arrows | Space/X stop | T takeoff | L land | +/- speed")

    def _add_event(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._recent_events.appendleft(f"[{ts}] {message}")

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _target_from_key(self, key: int) -> CommandState | None:
        # Instant target command increment, then smoothed into current cmd.
        step_v = 0.7
        step_yaw = 0.7

        target = CommandState(self.cmd.vx, self.cmd.vy, self.cmd.vz, self.cmd.yaw_rate_deg)

        if key in (ord("w"), ord("W"), curses.KEY_UP):
            target.vx += step_v
        elif key in (ord("s"), ord("S"), curses.KEY_DOWN):
            target.vx -= step_v
        elif key in (ord("a"), ord("A"), curses.KEY_LEFT):
            target.vy -= step_v
        elif key in (ord("d"), ord("D"), curses.KEY_RIGHT):
            target.vy += step_v
        elif key in (ord("r"), ord("R")):
            # NED: up means negative z velocity.
            target.vz -= step_v
        elif key in (ord("f"), ord("F")):
            target.vz += step_v
        elif key in (ord("q"), ord("Q")):
            target.yaw_rate_deg -= step_yaw
        elif key in (ord("e"), ord("E")):
            target.yaw_rate_deg += step_yaw
        elif key == ord(" "):
            target = CommandState()
        elif key in (ord("x"), ord("X")):
            target = CommandState()
        elif key in (ord("+"), ord("=")):
            self.max_speed = min(20.0, self.max_speed + 0.5)
            self._add_event(f"INFO max_speed -> {self.max_speed:.1f} m/s")
        elif key in (ord("-"), ord("_")):
            self.max_speed = max(0.5, self.max_speed - 0.5)
            self._add_event(f"INFO max_speed -> {self.max_speed:.1f} m/s")
        elif key in (ord("h"), ord("H")):
            self.print_help()
        elif key in (ord("t"), ord("T")):
            self._add_event("INFO takeoff")
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
            target = CommandState()
        elif key in (ord("l"), ord("L")):
            self._add_event("INFO land")
            self.client.landAsync(vehicle_name=self.vehicle_name).join()
            target = CommandState()
        elif key == 27:
            self.running = False
        else:
            return None

        target.vx = self._clamp(target.vx, -1.0, 1.0)
        target.vy = self._clamp(target.vy, -1.0, 1.0)
        target.vz = self._clamp(target.vz, -1.0, 1.0)
        target.yaw_rate_deg = self._clamp(target.yaw_rate_deg, -1.0, 1.0)
        return target

    def _smooth_update(self, target: CommandState | None) -> None:
        if target is None:
            # Natural decay when no key is pressed.
            target = CommandState(
                vx=self.cmd.vx * 0.92,
                vy=self.cmd.vy * 0.92,
                vz=self.cmd.vz * 0.92,
                yaw_rate_deg=self.cmd.yaw_rate_deg * 0.88,
            )

        self.cmd.vx = (1.0 - self.alpha) * self.cmd.vx + self.alpha * target.vx
        self.cmd.vy = (1.0 - self.alpha) * self.cmd.vy + self.alpha * target.vy
        self.cmd.vz = (1.0 - self.alpha) * self.cmd.vz + self.alpha * target.vz
        self.cmd.yaw_rate_deg = (
            (1.0 - self.alpha) * self.cmd.yaw_rate_deg + self.alpha * target.yaw_rate_deg
        )

    def _send_control(self) -> None:
        vx = self.cmd.vx * self.max_speed
        vy = self.cmd.vy * self.max_speed
        vz = self.cmd.vz * self.max_speed
        yaw_rate = self.cmd.yaw_rate_deg * self.max_yaw_rate_deg

        self.client.moveByVelocityAsync(
            vx=float(vx),
            vy=float(vy),
            vz=float(vz),
            duration=self.dt * 1.4,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate)),
            vehicle_name=self.vehicle_name,
        )

    def _status_snapshot(self) -> dict[str, object]:
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        kin = state.kinematics_estimated
        pos = kin.position
        vel = kin.linear_velocity
        speed = math.sqrt(vel.x_val ** 2 + vel.y_val ** 2 + vel.z_val ** 2)

        collision = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        has_collision = bool(collision.has_collided)

        if has_collision and not self.prev_collision:
            obj_name = getattr(collision, "object_name", "")
            obj_id = getattr(collision, "object_id", -1)
            pen = getattr(collision, "penetration_depth", 0.0)
            impact = getattr(collision, "impact_point", None)
            if impact is not None:
                impact_text = f"({impact.x_val:.2f}, {impact.y_val:.2f}, {impact.z_val:.2f})"
            else:
                impact_text = "(n/a)"
            self._add_event(
                "COLLISION "
                f"object='{obj_name}' id={obj_id} penetration={float(pen):.4f} impact={impact_text}"
            )

        self.prev_collision = has_collision
        return {
            "pos": pos,
            "vel": vel,
            "speed": speed,
            "has_collision": has_collision,
        }

    def _render_ui(self, stdscr: curses.window, snapshot: dict[str, object]) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()

        pos = snapshot["pos"]
        vel = snapshot["vel"]
        speed = float(snapshot["speed"])
        collision_flag = "YES" if bool(snapshot["has_collision"]) else "NO "

        lines = [
            "=== UAV Keyboard Teleop (Fixed Panel) ===",
            f"Vehicle: {self.vehicle_name}",
            "Keys: W/S A/D R/F Q/E | Arrows | Space/X stop | T takeoff | L land | +/- speed | Esc quit",
            "",
            f"Position (NED): x={pos.x_val:8.2f}  y={pos.y_val:8.2f}  z={pos.z_val:8.2f}",
            f"Velocity (m/s): vx={vel.x_val:7.2f}  vy={vel.y_val:7.2f}  vz={vel.z_val:7.2f}",
            f"Speed: {speed:6.2f} m/s   Collision: {collision_flag}",
            "",
            (
                "Command: "
                f"vx={self.cmd.vx*self.max_speed:7.2f}  "
                f"vy={self.cmd.vy*self.max_speed:7.2f}  "
                f"vz={self.cmd.vz*self.max_speed:7.2f}  "
                f"yaw_rate={self.cmd.yaw_rate_deg*self.max_yaw_rate_deg:7.2f} deg/s"
            ),
            f"Limit: max_speed={self.max_speed:5.2f} m/s   max_yaw_rate={self.max_yaw_rate_deg:5.1f} deg/s",
            "",
            "Recent Events:",
        ]

        for i in range(self._recent_events.maxlen):
            if i < len(self._recent_events):
                lines.append(f"  {self._recent_events[i]}")
            else:
                lines.append("  ")

        # curses handles line clipping by width; keep one-column margin for safety.
        max_col = max(1, width - 1)
        max_row = max(1, height - 1)
        for row, line in enumerate(lines):
            if row >= max_row:
                break
            stdscr.addnstr(row, 0, line, max_col)

        stdscr.noutrefresh()
        curses.doupdate()

    def run(self) -> None:
        self.print_help()
        self._add_event("INFO Teleop started. Press Esc to quit.")

        curses.wrapper(self._curses_loop)
        print("[INFO] Exiting teleop...")

    def _curses_loop(self, stdscr: curses.window) -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)

        next_tick = time.perf_counter()
        while self.running:
            key = stdscr.getch()
            target = self._target_from_key(key) if key != -1 else None
            self._smooth_update(target)
            self._send_control()
            snapshot = self._status_snapshot()
            self._render_ui(stdscr, snapshot)

            next_tick += self.dt
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyboard teleop for AirSim drone")
    parser.add_argument("--vehicle-name", type=str, default="Drone1", help="AirSim vehicle_name")
    parser.add_argument("--max-speed", type=float, default=6.0, help="Max translational speed (m/s)")
    parser.add_argument("--max-yaw-rate", type=float, default=60.0, help="Max yaw rate (deg/s)")
    parser.add_argument("--dt", type=float, default=0.05, help="Control loop period in seconds")
    parser.add_argument(
        "--no-auto-takeoff",
        action="store_true",
        help="Do not auto takeoff after connecting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    teleop = KeyboardTeleop(
        vehicle_name=args.vehicle_name,
        max_speed=float(args.max_speed),
        max_yaw_rate_deg=float(args.max_yaw_rate),
        dt=float(args.dt),
        auto_takeoff=not bool(args.no_auto_takeoff),
    )

    try:
        teleop.connect()
        teleop.run()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        teleop.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
