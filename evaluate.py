import argparse
import inspect
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required for evaluate.py. Install with: pip install pyyaml"
    ) from exc

from gymnasium import spaces
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from envs import UAVSimpleTrainEnv
from utils.pace_planner import (
    AStarPlanner,
    AStarPlanner3D,
    PixelToAirSimMatrixMapper,
    sparsify_path,
)

Pixel = Tuple[int, int]
Waypoint = Tuple[float, float, float]
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"


@dataclass
class PlannerConfig:
    map_path: str
    pixel_to_airsim_matrix: str
    obstacle_height_threshold: float = 25.0
    height_channel: str = "r"
    waypoint_stride: int = 15
    planning_mode: str = "3d"
    height_weight: float = 8.0
    height_lift_coef: float = 0.3
    max_segment_altitude_delta: Optional[float] = 4.0


@dataclass
class PathsConfig:
    paths_json: str
    height: Optional[float] = None
    ned: Optional[bool] = None


@dataclass
class EnvConfig:
    max_speed: Optional[float] = None
    step_duration: Optional[float] = None
    max_episode_steps: Optional[int] = None
    success_threshold: Optional[float] = 0.0
    depth_feature_size: Optional[Tuple[int, int]] = None
    max_depth_m: Optional[float] = None
    spawn_points_json: Optional[str] = None
    goal_distance_range: Optional[Tuple[float, float]] = None


@dataclass
class EvalConfig:
    num_episodes: int
    model_path: str
    model_algo: str = "TD3"
    deterministic: bool = True
    max_steps_per_episode: int = 1000
    waypoint_tolerance: float = 2.0
    output_json: str = "eval_results.json"
    log_level: str = "INFO"
    seed: int = 42
    shuffle_paths: bool = False
    visualize: bool = False
    visualize_step_interval: int = 1
    visualize_pause_sec: float = 0.0
    visualize_window_scale: float = 1.0
    interactive_select: bool = False
    interactive_vertical_speed: float = 1.0
    interactive_takeoff_clearance: float = 0.5
    interactive_landing_clearance: float = 1.5
    interactive_landing_timeout_sec: float = 60.0


@dataclass
class EvaluationConfig:
    planner: PlannerConfig
    paths: PathsConfig
    env: EnvConfig
    eval: EvalConfig


@dataclass
class PathPair:
    start: np.ndarray
    end: np.ndarray
    path_id: Optional[int] = None
    distance: Optional[float] = None


@dataclass
class WaypointPlan:
    waypoints: List[np.ndarray]
    ideal_length: float
    start_pixel: Pixel
    end_pixel: Pixel
    pixel_path: List[Pixel]
    raw_altitude_range: Optional[Tuple[float, float]] = None
    inserted_transition_waypoint_count: int = 0


@dataclass
class InteractiveFlightProfile:
    launch_position: np.ndarray
    landing_position: np.ndarray
    launch_surface_z: float
    vertical_speed: float = 1.0
    landing_approach_clearance: float = 1.5
    landing_timeout_sec: float = 60.0


@dataclass
class EpisodeResult:
    episode_id: int
    path_id: Optional[int]
    success: bool
    collision: bool
    timeout: bool
    planning_failed: bool
    failed: bool
    steps: int
    duration_sec: float
    ideal_path_length: float
    actual_path_length: float
    spl: float
    waypoint_count: int
    completed_waypoints: int
    start: List[float]
    end: List[float]
    final_position: Optional[List[float]]
    final_distance_to_goal: Optional[float]
    reason: str


class BaseLocalNavigator(ABC):
    """Abstract navigator interface for local policy inference."""

    @abstractmethod
    def predict(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SB3LocalNavigator(BaseLocalNavigator):
    """Stable Baselines3 model wrapper with a unified predict interface."""

    def __init__(
        self,
        model_path: str,
        algo: str,
        deterministic: bool,
    ) -> None:
        self.model_path = model_path
        self.algo = algo.upper()
        self.deterministic = deterministic
        self.model = self._load_model()

    def _load_model(self) -> BaseAlgorithm:
        model_cls = self._get_algorithm_class(self.algo)
        return model_cls.load(self.model_path)

    @staticmethod
    def _get_algorithm_class(name: str) -> Type[BaseAlgorithm]:
        if name == "TD3":
            return TD3
        if name == "SAC":
            return SAC
        raise ValueError(f"Unsupported SB3 algorithm: {name}")

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)


class GlobalPlanner:
    """Compute global A* waypoints using a height-map grid and homography."""

    def __init__(
        self,
        planner: AStarPlanner,
        mapper: PixelToAirSimMatrixMapper,
        waypoint_stride: int,
        z_up: float,
        planning_mode: str = "3d",
        height_lift_coef: float = 0.3,
        max_segment_altitude_delta: Optional[float] = 4.0,
        allow_diagonal: bool = True,
        prevent_corner_cutting: bool = True,
    ) -> None:
        self.planner = planner
        self.mapper = mapper
        self.waypoint_stride = max(1, int(waypoint_stride))
        self.z_up = float(z_up)
        self.planning_mode = str(planning_mode).lower()
        self.height_lift_coef = float(height_lift_coef)
        self.max_segment_altitude_delta = (
            None
            if max_segment_altitude_delta is None
            else float(max_segment_altitude_delta)
        )
        if (
            self.max_segment_altitude_delta is not None
            and self.max_segment_altitude_delta <= 0.0
        ):
            raise ValueError("max_segment_altitude_delta must be > 0 or null")
        self.allow_diagonal = bool(allow_diagonal)
        self.prevent_corner_cutting = bool(prevent_corner_cutting)

    def plan(self, start: np.ndarray, end: np.ndarray) -> WaypointPlan:
        start_pixel = self._airsim_to_pixel(start)
        end_pixel = self._airsim_to_pixel(end)
        inserted_transition_waypoint_count = 0
        raw_altitude_range: Optional[Tuple[float, float]] = None

        if self.planning_mode == "3d":
            if not isinstance(self.planner, AStarPlanner3D):
                raise TypeError("3D planning requires AStarPlanner3D")
            pixel_path_3d = self.planner.astar_3d(
                start_pixel,
                end_pixel,
                cruise_alt=self.z_up,
                height_lift_coef=self.height_lift_coef,
                allow_diagonal=self.allow_diagonal,
                prevent_corner_cutting=self.prevent_corner_cutting,
            )
            if not pixel_path_3d:
                raise RuntimeError("3D A* returned empty path")
            raw_z_values = [float(point[2]) for point in pixel_path_3d]
            raw_altitude_range = (min(raw_z_values), max(raw_z_values))
            pixel_path_3d_sparse, inserted_transition_waypoint_count = (
                self._sparsify_3d_path(pixel_path_3d)
            )
            pixel_path_sparse = [(p[0], p[1]) for p in pixel_path_3d_sparse]
            waypoints = [
                np.array(
                    (*self.mapper.pixel_to_airsim_xy((u, v)), float(z)),
                    dtype=np.float32,
                )
                for u, v, z in pixel_path_3d_sparse
            ]
        elif self.planning_mode == "2d":
            pixel_path = self.planner.astar(
                start_pixel,
                end_pixel,
                allow_diagonal=self.allow_diagonal,
                prevent_corner_cutting=self.prevent_corner_cutting,
            )
            if not pixel_path:
                raise RuntimeError("2D A* returned empty path")
            pixel_path_sparse = sparsify_path(pixel_path, self.waypoint_stride)
            waypoints = [
                np.array(
                    self.mapper.pixel_to_waypoint(pixel, self.z_up),
                    dtype=np.float32,
                )
                for pixel in pixel_path_sparse
            ]
            raw_altitude_range = (self.z_up, self.z_up)
        else:
            raise ValueError(f"Unsupported planning mode: {self.planning_mode}")

        waypoints[0] = start.astype(np.float32)
        waypoints[-1] = end.astype(np.float32)
        if self.planning_mode == "3d":
            waypoints, synthetic_count = self._insert_altitude_transitions(waypoints)
            inserted_transition_waypoint_count += synthetic_count

        ideal_length = compute_waypoint_length(waypoints)
        return WaypointPlan(
            waypoints=waypoints,
            ideal_length=ideal_length,
            start_pixel=start_pixel,
            end_pixel=end_pixel,
            pixel_path=pixel_path_sparse,
            raw_altitude_range=raw_altitude_range,
            inserted_transition_waypoint_count=inserted_transition_waypoint_count,
        )

    def _airsim_to_pixel(self, position: np.ndarray) -> Pixel:
        u, v = self.mapper.airsim_xy_to_pixel((float(position[0]), float(position[1])))
        return int(round(u)), int(round(v))

    def _sparsify_3d_path(
        self, pixel_path: Sequence[Tuple[int, int, float]]
    ) -> Tuple[List[Tuple[int, int, float]], int]:
        """Keep route altitude extrema and intermediate points needed for gradual changes."""
        if not pixel_path:
            return [], 0

        baseline_indices = set(range(0, len(pixel_path), self.waypoint_stride))
        baseline_indices.add(len(pixel_path) - 1)
        selected_indices = set(baseline_indices)
        ordered_baseline = sorted(baseline_indices)

        # Preserve the highest and lowest planned altitudes inside every sparse span.
        for start_idx, end_idx in zip(ordered_baseline, ordered_baseline[1:]):
            interval = range(start_idx, end_idx + 1)
            selected_indices.add(min(interval, key=lambda idx: pixel_path[idx][2]))
            selected_indices.add(max(interval, key=lambda idx: pixel_path[idx][2]))

        if self.max_segment_altitude_delta is not None:
            altitude_anchor_idx = 0
            for idx in range(1, len(pixel_path)):
                delta_z = abs(
                    pixel_path[idx][2] - pixel_path[altitude_anchor_idx][2]
                )
                if delta_z >= self.max_segment_altitude_delta:
                    previous_idx = idx - 1
                    if (
                        delta_z > self.max_segment_altitude_delta
                        and previous_idx > altitude_anchor_idx
                        and not np.isclose(
                            pixel_path[previous_idx][2],
                            pixel_path[altitude_anchor_idx][2],
                        )
                    ):
                        selected_indices.add(previous_idx)
                        altitude_anchor_idx = previous_idx
                    selected_indices.add(idx)
                    altitude_anchor_idx = idx

        selected_order = sorted(selected_indices)
        inserted_count = len(selected_indices - baseline_indices)
        return [pixel_path[idx] for idx in selected_order], inserted_count

    def _insert_altitude_transitions(
        self, waypoints: Sequence[np.ndarray]
    ) -> Tuple[List[np.ndarray], int]:
        """Insert climb-before-travel or descend-after-travel transition goals."""
        if self.max_segment_altitude_delta is None or len(waypoints) < 2:
            return list(waypoints), 0

        max_delta = self.max_segment_altitude_delta
        transitioned = [np.asarray(waypoints[0], dtype=np.float32)]
        inserted_count = 0

        for target_value in waypoints[1:]:
            target = np.asarray(target_value, dtype=np.float32)
            source = transitioned[-1]
            delta_z = float(target[2] - source[2])

            if abs(delta_z) <= max_delta:
                transitioned.append(target)
                continue

            if delta_z > 0.0 and not np.allclose(source[:2], target[:2]):
                approach = np.array([target[0], target[1], source[2]], dtype=np.float32)
                transitioned.append(approach)
                inserted_count += 1
                source = approach
                delta_z = float(target[2] - source[2])

            transition_xy = source[:2] if delta_z < 0.0 else target[:2]
            direction = -1.0 if delta_z < 0.0 else 1.0
            transition_z = float(source[2])
            while abs(float(target[2]) - transition_z) > max_delta:
                transition_z += direction * max_delta
                transitioned.append(
                    np.array(
                        [transition_xy[0], transition_xy[1], transition_z],
                        dtype=np.float32,
                    )
                )
                inserted_count += 1

            transitioned.append(target)

        return transitioned, inserted_count


class EpisodeExecutor:
    """Run a single hierarchical navigation episode without resetting per waypoint."""

    def __init__(
        self,
        env: UAVSimpleTrainEnv,
        navigator: BaseLocalNavigator,
        waypoint_tolerance: float,
        max_steps: int,
        logger: logging.Logger,
        visualizer: Optional["MapVisualizer"] = None,
    ) -> None:
        self.env = env
        self.navigator = navigator
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.max_steps = int(max_steps)
        self.logger = logger
        self.visualizer = visualizer

    def run_episode(
        self,
        episode_id: int,
        path_pair: PathPair,
        plan: WaypointPlan,
        interactive_profile: Optional[InteractiveFlightProfile] = None,
    ) -> EpisodeResult:
        start_time = time.time()
        initial_position = (
            interactive_profile.launch_position
            if interactive_profile is not None
            else path_pair.start
        )
        self.env.start_pos = initial_position.astype(np.float32)
        reset_obs, _ = parse_reset_output(self.env.reset())

        current_wp_idx = 0
        completed_waypoints = 0
        steps = 0
        actual_length = 0.0
        prev_pos = get_current_position_from_env(self.env)
        collision = False
        success = False
        timeout = False
        reason = "running"
        phase_failed = False

        if self.visualizer is not None:
            self.visualizer.start_episode(path_pair.start, path_pair.end)
            self.visualizer.update_position(prev_pos, 0)

        obs = reset_obs
        if interactive_profile is not None:
            try:
                obs, takeoff_pos = self._execute_interactive_takeoff(
                    episode_id=episode_id,
                    target=path_pair.start,
                    profile=interactive_profile,
                    obs=obs,
                    current_pos=prev_pos,
                )
                actual_length += float(np.linalg.norm(takeoff_pos - prev_pos))
                prev_pos = takeoff_pos
            except Exception as exc:
                self.logger.error("Episode %d takeoff failed: %s", episode_id, exc)
                phase_failed = True
                reason = "takeoff_failed"

        if not phase_failed:
            current_wp_idx, completed_waypoints, obs, success, _ = (
                self._advance_reached_waypoints(
                    episode_id=episode_id,
                    plan=plan,
                    current_pos=prev_pos,
                    current_wp_idx=current_wp_idx,
                    completed_waypoints=completed_waypoints,
                    obs=obs,
                    activate_target=True,
                )
            )
            if success:
                reason = "success"

        while not phase_failed and not success and steps < self.max_steps:
            action = self.navigator.predict(obs)
            step_out = self.env.step(action)
            obs, _reward, terminated, truncated, info = parse_step_output(step_out)
            steps += 1

            current_pos = get_current_position_from_env(self.env)
            actual_length += float(np.linalg.norm(current_pos - prev_pos))
            prev_pos = current_pos

            if self.visualizer is not None:
                self.visualizer.update_position(current_pos, steps)

            collision = bool(info.get("collision", False))
            if collision:
                reason = "collision"
                break

            current_wp_idx, completed_waypoints, obs, success, advanced = (
                self._advance_reached_waypoints(
                    episode_id=episode_id,
                    plan=plan,
                    current_pos=current_pos,
                    current_wp_idx=current_wp_idx,
                    completed_waypoints=completed_waypoints,
                    obs=obs,
                    activate_target=False,
                )
            )
            if success:
                reason = "success"
                break

            # The current environment terminates on local-target arrival or collision.
            # Continue only when this step demonstrably reached and switched a waypoint.
            if terminated and not advanced:
                reason = "env_terminated"
                break

            if truncated:
                timeout = True
                reason = "env_truncated"
                break

        if not phase_failed and not (success or collision or timeout) and steps >= self.max_steps:
            timeout = True
            reason = "max_steps"

        if success and interactive_profile is not None:
            try:
                landing_pos = self._execute_interactive_landing(
                    episode_id=episode_id,
                    profile=interactive_profile,
                    current_pos=prev_pos,
                )
                actual_length += float(np.linalg.norm(landing_pos - prev_pos))
                prev_pos = landing_pos
                reason = "success_landed"
            except Exception as exc:
                self.logger.error("Episode %d landing failed: %s", episode_id, exc)
                success = False
                reason = "landing_failed"

        duration_sec = time.time() - start_time
        ideal_length = plan.ideal_length
        metric_goal = plan.waypoints[-1]
        if interactive_profile is not None:
            ideal_length += float(
                np.linalg.norm(path_pair.start - interactive_profile.launch_position)
            )
            ideal_length += float(
                np.linalg.norm(plan.waypoints[-1] - interactive_profile.landing_position)
            )
            metric_goal = interactive_profile.landing_position
        spl = compute_spl(success, ideal_length, actual_length)
        final_distance_to_goal = float(np.linalg.norm(prev_pos - metric_goal))
        failed = not success

        result = EpisodeResult(
            episode_id=episode_id,
            path_id=path_pair.path_id,
            success=success,
            collision=collision,
            timeout=timeout,
            planning_failed=False,
            failed=failed,
            steps=steps,
            duration_sec=duration_sec,
            ideal_path_length=ideal_length,
            actual_path_length=actual_length,
            spl=spl,
            waypoint_count=len(plan.waypoints),
            completed_waypoints=completed_waypoints,
            start=path_pair.start.tolist(),
            end=path_pair.end.tolist(),
            final_position=prev_pos.tolist() if isinstance(prev_pos, np.ndarray) else None,
            final_distance_to_goal=final_distance_to_goal,
            reason=reason,
        )
        return result

    def _execute_interactive_takeoff(
        self,
        episode_id: int,
        target: np.ndarray,
        profile: InteractiveFlightProfile,
        obs: np.ndarray,
        current_pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if float(target[2]) >= float(profile.launch_position[2]):
            raise ValueError(
                "Selected takeoff height is not above the estimated launch surface"
            )

        self.logger.info(
            "Episode %d vertical takeoff: surface_z=%.2f, target_z=%.2f, speed=%.2f m/s",
            episode_id,
            profile.launch_surface_z,
            float(target[2]),
            profile.vertical_speed,
        )
        obs = update_env_target(self.env, target, obs, segment_start=current_pos)
        self._move_vertical_to_z(float(target[2]), profile.vertical_speed)
        takeoff_pos = get_current_position_from_env(self.env)
        if self.visualizer is not None:
            self.visualizer.update_position(takeoff_pos, 0)
        self.logger.info(
            "Episode %d takeoff complete: position=(%.2f, %.2f, %.2f)",
            episode_id,
            float(takeoff_pos[0]),
            float(takeoff_pos[1]),
            float(takeoff_pos[2]),
        )
        return obs, takeoff_pos

    def _execute_interactive_landing(
        self,
        episode_id: int,
        profile: InteractiveFlightProfile,
        current_pos: np.ndarray,
    ) -> np.ndarray:
        approach_z = float(profile.landing_position[2] - profile.landing_approach_clearance)
        self.logger.info(
            "Episode %d landing approach: estimated_surface_z=%.2f, approach_z=%.2f",
            episode_id,
            float(profile.landing_position[2]),
            approach_z,
        )
        if float(current_pos[2]) < approach_z:
            self._move_vertical_to_z(approach_z, profile.vertical_speed)
        elif float(current_pos[2]) > approach_z + self.waypoint_tolerance:
            self.logger.warning(
                "Episode %d is below estimated landing approach height; "
                "delegating final contact directly to AirSim landing control.",
                episode_id,
            )

        client = getattr(self.env, "client", None)
        if client is None or not hasattr(client, "landAsync"):
            raise RuntimeError("Environment client does not expose landAsync")
        if hasattr(client, "hoverAsync"):
            client.hoverAsync().join()
        client.landAsync(timeout_sec=profile.landing_timeout_sec).join()
        landing_pos = get_current_position_from_env(self.env)
        if self.visualizer is not None:
            self.visualizer.update_position(landing_pos, 0)
        self.logger.info(
            "Episode %d landed: position=(%.2f, %.2f, %.2f), estimated_surface_z=%.2f",
            episode_id,
            float(landing_pos[0]),
            float(landing_pos[1]),
            float(landing_pos[2]),
            float(profile.landing_position[2]),
        )
        return landing_pos

    def _move_vertical_to_z(self, target_z: float, speed: float) -> None:
        client = getattr(self.env, "client", None)
        if client is None or not hasattr(client, "moveToZAsync"):
            raise RuntimeError("Environment client does not expose moveToZAsync")
        if hasattr(client, "hoverAsync"):
            client.hoverAsync().join()
        client.moveToZAsync(float(target_z), max(0.1, float(speed))).join()
        if hasattr(client, "hoverAsync"):
            client.hoverAsync().join()

    def _advance_reached_waypoints(
        self,
        episode_id: int,
        plan: WaypointPlan,
        current_pos: np.ndarray,
        current_wp_idx: int,
        completed_waypoints: int,
        obs: np.ndarray,
        activate_target: bool,
    ) -> Tuple[int, int, np.ndarray, bool, bool]:
        advanced = False
        while current_wp_idx < len(plan.waypoints):
            waypoint = plan.waypoints[current_wp_idx]
            distance_to_waypoint = float(np.linalg.norm(current_pos - waypoint))
            if distance_to_waypoint > self.waypoint_tolerance:
                if activate_target:
                    obs = update_env_target(
                        self.env, waypoint, obs, segment_start=current_pos
                    )
                return current_wp_idx, completed_waypoints, obs, False, advanced

            completed_waypoints += 1
            advanced = True
            if current_wp_idx >= len(plan.waypoints) - 1:
                return current_wp_idx, completed_waypoints, obs, True, advanced

            next_wp_idx = current_wp_idx + 1
            self.logger.info(
                "Episode %d reached waypoint %d/%d (dist=%.2f), switching to %d/%d",
                episode_id,
                current_wp_idx + 1,
                len(plan.waypoints),
                distance_to_waypoint,
                next_wp_idx + 1,
                len(plan.waypoints),
            )
            current_wp_idx = next_wp_idx
            activate_target = True

        return current_wp_idx, completed_waypoints, obs, True, advanced


class MetricsTracker:
    """Aggregate per-episode metrics and compute summary statistics."""

    def __init__(self) -> None:
        self.results: List[EpisodeResult] = []

    def add(self, result: EpisodeResult) -> None:
        self.results.append(result)

    def summary(self) -> Dict[str, Any]:
        total = len(self.results)
        if total == 0:
            return {
                "total_episodes": 0,
                "success_rate": 0.0,
                "collision_rate": 0.0,
                "timeout_rate": 0.0,
                "spl": 0.0,
                "avg_steps_success": 0.0,
                "avg_time_sec_success": 0.0,
                "failed_count": 0,
                "avg_failed_distance": 0.0,
            }

        successes = [r for r in self.results if r.success]
        collisions = [r for r in self.results if r.collision]
        timeouts = [r for r in self.results if r.timeout]
        failed = [r for r in self.results if r.failed]

        success_rate = len(successes) / total
        collision_rate = len(collisions) / total
        timeout_rate = len(timeouts) / total

        spl_values = [r.spl for r in self.results]
        avg_spl = float(np.mean(spl_values)) if spl_values else 0.0

        avg_steps_success = (
            float(np.mean([r.steps for r in successes])) if successes else 0.0
        )
        avg_time_sec_success = (
            float(np.mean([r.duration_sec for r in successes])) if successes else 0.0
        )
        failed_distances = [
            r.final_distance_to_goal
            for r in failed
            if r.final_distance_to_goal is not None
        ]
        avg_failed_distance = (
            float(np.mean(failed_distances)) if failed_distances else 0.0
        )

        return {
            "total_episodes": total,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "timeout_rate": timeout_rate,
            "spl": avg_spl,
            "avg_steps_success": avg_steps_success,
            "avg_time_sec_success": avg_time_sec_success,
            "failed_count": len(failed),
            "avg_failed_distance": avg_failed_distance,
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "episodes": [episode_to_dict(r) for r in self.results],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hierarchical UAV navigation")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to YAML config file.",
    )
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-algo", type=str, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--waypoint-tolerance", type=float, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle-paths", action="store_true")
    parser.add_argument("--map-path", type=str, default=None)
    parser.add_argument("--matrix-path", type=str, default=None)
    parser.add_argument("--paths-json", type=str, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--obstacle-height-threshold", type=float, default=None)
    parser.add_argument("--height-channel", type=str, default=None)
    parser.add_argument("--planning-mode", type=str, choices=("2d", "3d"), default=None)
    parser.add_argument("--height-weight", type=float, default=None)
    parser.add_argument("--height-lift-coef", type=float, default=None)
    parser.add_argument("--max-segment-altitude-delta", type=float, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--interactive-select", action="store_true")
    return parser.parse_args()


class VisualizerLogHandler(logging.Handler):
    """Mirror formatted application log records into a map visualizer."""

    def __init__(self, visualizer: "MapVisualizer") -> None:
        super().__init__()
        self.visualizer = visualizer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.visualizer.append_log(self.format(record))
        except Exception:
            self.handleError(record)


class MapVisualizer:
    """Render an OpenCV-drawn map inside a Tkinter visualization window."""

    MAX_LOG_LINES = 500
    WINDOW_SCREEN_MARGIN = 80
    CONTROL_PANEL_RESERVED_HEIGHT = 270

    def __init__(
        self,
        map_path: str,
        mapper: PixelToAirSimMatrixMapper,
        step_interval: int,
        pause_sec: float,
        window_scale: float,
        logger: logging.Logger,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for visualization. Install opencv-python."
            ) from exc
        try:
            import tkinter as tk
        except ImportError as exc:
            raise ImportError(
                "Tkinter is required for visualization and interactive selection."
            ) from exc

        self._cv2 = cv2
        self._tk = tk
        self.map_path = map_path
        self.mapper = mapper
        self.step_interval = max(1, int(step_interval))
        self.pause_ms = max(1, int(float(pause_sec) * 1000.0))
        self.window_scale = max(0.1, float(window_scale))
        self.logger = logger

        raw_img = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            raise ValueError(f"Cannot read map image: {map_path}")

        if raw_img.ndim == 2:
            base = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
        elif raw_img.shape[2] == 4:
            base = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2BGR)
        else:
            base = raw_img.copy()

        self.base_map = base
        self.window_name = "UAV Navigation Map"
        self.current_map = base.copy()
        self.prev_pixel: Optional[Tuple[int, int]] = None
        self.log_handler: Optional[VisualizerLogHandler] = None
        self.closed = False
        self.selection_active = False
        self.selection_confirmed = False
        self.selection_cancelled = False
        self.selection_points: List[Tuple[int, int]] = []
        self._photo_image = None

        self.root = tk.Tk()
        self.root.title(self.window_name)
        self.root.protocol("WM_DELETE_WINDOW", self._cancel_or_close)
        self.root.bind("<Escape>", lambda _event: self._cancel_or_close())
        self.render_width, self.render_height = self._resolve_render_size()

        self.map_label = tk.Label(self.root, bg="black", bd=0)
        self.map_label.pack(side=tk.TOP, fill=tk.NONE)
        self.map_label.bind("<Button-1>", self._on_map_left_click)
        self.map_label.bind("<Button-3>", self._on_map_right_click)

        self.action_frame = tk.Frame(self.root, padx=8, pady=6)
        self.action_frame.pack(side=tk.TOP, fill=tk.X)
        self.hint_label = tk.Label(
            self.action_frame,
            text="Trajectory visualization",
            anchor="w",
        )
        self.hint_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.confirm_button = tk.Button(
            self.action_frame,
            text="Confirm",
            width=12,
            command=self._confirm_selection,
            state=tk.DISABLED,
        )

        log_frame = tk.LabelFrame(self.root, text="Log", padx=4, pady=4)
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.log_text = tk.Text(
            log_frame,
            height=10,
            wrap=tk.NONE,
            state=tk.DISABLED,
            bg="#191919",
            fg="#dddddd",
            insertbackground="#dddddd",
        )
        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._render()

    def enable_log_mirroring(self) -> None:
        if self.log_handler is not None:
            return

        handler = VisualizerLogHandler(self)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        self.logger.addHandler(handler)
        self.log_handler = handler

    def append_log(self, message: str) -> None:
        if self.closed:
            return
        self.log_text.configure(state=self._tk.NORMAL)
        self.log_text.insert(self._tk.END, f"{message}\n")
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > self.MAX_LOG_LINES:
            excess = line_count - self.MAX_LOG_LINES
            self.log_text.delete("1.0", f"{excess + 1}.0")
        self.log_text.see(self._tk.END)
        self.log_text.configure(state=self._tk.DISABLED)
        self._process_gui_events()

    def start_episode(self, start: np.ndarray, end: np.ndarray) -> None:
        self.current_map = self.base_map.copy()
        self.prev_pixel = None

        start_pixel = self._airsim_to_pixel(start)
        end_pixel = self._airsim_to_pixel(end)

        self._draw_marker(start_pixel, (0, 255, 0), "S")
        self._draw_marker(end_pixel, (0, 255, 0), "G")
        self._render()

    def select_start_goal(self, z_value: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        self.selection_active = True
        self.selection_confirmed = False
        self.selection_cancelled = False
        self.selection_points.clear()
        self.confirm_button.configure(state=self._tk.DISABLED)
        self.confirm_button.pack(side=self._tk.RIGHT, padx=(8, 0))
        self.hint_label.configure(
            text="Left click: select start/goal    Right click: reset    Esc: cancel"
        )
        self.logger.info(
            "Interactive selection: left click start/goal, right click reset, "
            "then click Confirm."
        )

        while not (self.selection_confirmed or self.selection_cancelled or self.closed):
            self.current_map = self.base_map.copy()
            if self.selection_points:
                self._draw_marker(self.selection_points[0], (0, 255, 0), "S")
            if len(self.selection_points) >= 2:
                self._draw_marker(self.selection_points[1], (0, 255, 0), "G")
            self._render()
            time.sleep(self.pause_ms / 1000.0)

        self.selection_active = False
        self.confirm_button.pack_forget()
        self.hint_label.configure(text="Trajectory visualization")
        if self.selection_cancelled or self.closed:
            return None

        start = self._pixel_to_airsim(self.selection_points[0], z_value)
        end = self._pixel_to_airsim(self.selection_points[1], z_value)
        return start, end

    def update_position(self, position: np.ndarray, step: int) -> None:
        if step % self.step_interval != 0:
            return

        pixel = self._airsim_to_pixel(position)
        if self.prev_pixel is None:
            self.prev_pixel = pixel
        else:
            self._cv2.line(
                self.current_map,
                self.prev_pixel,
                pixel,
                color=(255, 255, 0),
                thickness=3,
                lineType=self._cv2.LINE_AA,
            )
            self.prev_pixel = pixel

        self._render()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.log_handler is not None:
            self.logger.removeHandler(self.log_handler)
            self.log_handler.close()
            self.log_handler = None
        try:
            self.root.destroy()
        except Exception:
            pass

    def wait_until_closed(self) -> None:
        if self.closed:
            return
        self.hint_label.configure(text="Execution finished. Close this window to exit.")
        self._process_gui_events()
        while not self.closed:
            self._process_gui_events()
            time.sleep(self.pause_ms / 1000.0)

    def _render(self) -> None:
        if self.closed:
            return
        rgb_image = self._image_to_rgb8(self.current_map)
        if rgb_image.shape[:2] != (self.render_height, self.render_width):
            rgb_image = self._cv2.resize(
                rgb_image,
                (self.render_width, self.render_height),
                interpolation=self._cv2.INTER_AREA,
            )
        header = f"P6 {self.render_width} {self.render_height} 255\n".encode("ascii")
        self._photo_image = self._tk.PhotoImage(
            data=header + rgb_image.tobytes(), format="PPM"
        )
        self.map_label.configure(image=self._photo_image)
        self._process_gui_events()

    def _resolve_render_size(self) -> Tuple[int, int]:
        requested_width = max(100, int(self.base_map.shape[1] * self.window_scale))
        requested_height = max(100, int(self.base_map.shape[0] * self.window_scale))
        max_width = max(100, int(self.root.winfo_screenwidth()) - self.WINDOW_SCREEN_MARGIN)
        max_height = max(
            100,
            int(self.root.winfo_screenheight())
            - self.WINDOW_SCREEN_MARGIN
            - self.CONTROL_PANEL_RESERVED_HEIGHT,
        )
        fit_scale = min(
            1.0,
            max_width / requested_width,
            max_height / requested_height,
        )
        return (
            max(100, int(requested_width * fit_scale)),
            max(100, int(requested_height * fit_scale)),
        )

    def _image_to_rgb8(self, image: np.ndarray) -> np.ndarray:
        if np.issubdtype(image.dtype, np.floating):
            bgr = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        elif image.dtype == np.uint16:
            bgr = (image / 256).astype(np.uint8)
        else:
            bgr = np.clip(image, 0, 255).astype(np.uint8)
        return self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)

    def _process_gui_events(self) -> None:
        if self.closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except self._tk.TclError:
            self.closed = True

    def _airsim_to_pixel(self, position: np.ndarray) -> Tuple[int, int]:
        u, v = self.mapper.airsim_xy_to_pixel((float(position[0]), float(position[1])))
        return int(round(u)), int(round(v))

    def _draw_marker(self, pixel: Tuple[int, int], color: Tuple[int, int, int], label: str) -> None:
        self._cv2.circle(self.current_map, pixel, radius=6, color=color, thickness=-1)
        self._cv2.putText(
            self.current_map,
            label,
            (pixel[0] + 6, pixel[1] - 6),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            self._cv2.LINE_AA,
        )

    def _on_map_left_click(self, event: Any) -> None:
        if not self.selection_active or len(self.selection_points) >= 2:
            return
        pixel = self._display_to_image_pixel(int(event.x), int(event.y))
        if pixel is None:
            return
        self.selection_points.append(pixel)
        if len(self.selection_points) == 2:
            self.confirm_button.configure(state=self._tk.NORMAL)

    def _on_map_right_click(self, _event: Any) -> None:
        if not self.selection_active:
            return
        self.selection_points.clear()
        self.selection_confirmed = False
        self.confirm_button.configure(state=self._tk.DISABLED)

    def _display_to_image_pixel(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        if not (0 <= x < self.render_width and 0 <= y < self.render_height):
            return None
        img_x = self._scale_display_axis_to_image(
            x, self.render_width, self.base_map.shape[1]
        )
        img_y = self._scale_display_axis_to_image(
            y, self.render_height, self.base_map.shape[0]
        )
        return img_x, img_y

    @staticmethod
    def _scale_display_axis_to_image(
        coordinate: int, display_size: int, image_size: int
    ) -> int:
        if display_size <= 1 or image_size <= 1:
            return 0
        scaled = coordinate * (image_size - 1) / (display_size - 1)
        return max(0, min(int(round(scaled)), image_size - 1))

    def _confirm_selection(self) -> None:
        if self.selection_active and len(self.selection_points) >= 2:
            self.selection_confirmed = True

    def _cancel_or_close(self) -> None:
        if self.selection_active:
            self.selection_cancelled = True
        else:
            self.close()

    def _pixel_to_airsim(self, pixel: Tuple[int, int], z_value: float) -> np.ndarray:
        x_air, y_air = self.mapper.pixel_to_airsim_xy(pixel)
        return np.array([x_air, y_air, float(z_value)], dtype=np.float32)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config YAML must be a mapping at the top level")
    return payload


def resolve_path(base_dir: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def load_config(args: argparse.Namespace) -> EvaluationConfig:
    if not args.config:
        raise ValueError("--config is required for YAML configuration")

    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    payload = load_yaml(config_path)

    planner_payload = payload.get("planner", {})
    paths_payload = payload.get("paths", {})
    env_payload = payload.get("env", {})
    eval_payload = payload.get("eval", {})

    map_path = args.map_path or planner_payload.get("map_path")
    matrix_path = args.matrix_path or planner_payload.get("pixel_to_airsim_matrix")
    paths_json = args.paths_json or paths_payload.get("paths_json")

    if map_path is None or matrix_path is None or paths_json is None:
        raise ValueError("planner.map_path, planner.pixel_to_airsim_matrix, and paths.paths_json are required")

    planning_mode = str(
        args.planning_mode
        if args.planning_mode is not None
        else planner_payload.get("planning_mode", "3d")
    ).lower()
    if planning_mode not in ("2d", "3d"):
        raise ValueError("planner.planning_mode must be '2d' or '3d'")

    max_segment_altitude_delta = (
        args.max_segment_altitude_delta
        if args.max_segment_altitude_delta is not None
        else planner_payload.get("max_segment_altitude_delta", 4.0)
    )
    if max_segment_altitude_delta is not None:
        max_segment_altitude_delta = float(max_segment_altitude_delta)
        if max_segment_altitude_delta <= 0.0:
            raise ValueError("planner.max_segment_altitude_delta must be > 0 or null")

    planner = PlannerConfig(
        map_path=resolve_path(config_dir, map_path),
        pixel_to_airsim_matrix=resolve_path(config_dir, matrix_path),
        obstacle_height_threshold=float(
            args.obstacle_height_threshold
            if args.obstacle_height_threshold is not None
            else planner_payload.get("obstacle_height_threshold", 25.0)
        ),
        height_channel=str(
            args.height_channel
            if args.height_channel is not None
            else planner_payload.get("height_channel", "r")
        ),
        waypoint_stride=int(planner_payload.get("waypoint_stride", 6)),
        planning_mode=planning_mode,
        height_weight=float(
            args.height_weight
            if args.height_weight is not None
            else planner_payload.get("height_weight", 8.0)
        ),
        height_lift_coef=float(
            args.height_lift_coef
            if args.height_lift_coef is not None
            else planner_payload.get("height_lift_coef", 0.3)
        ),
        max_segment_altitude_delta=max_segment_altitude_delta,
    )

    paths = PathsConfig(
        paths_json=resolve_path(config_dir, paths_json),
        height=float(args.height)
        if args.height is not None
        else (paths_payload.get("height") if paths_payload.get("height") is not None else None),
        ned=paths_payload.get("ned"),
    )

    depth_feature = env_payload.get("depth_feature_size")
    depth_feature_tuple: Optional[Tuple[int, int]] = None
    if isinstance(depth_feature, (list, tuple)) and len(depth_feature) == 2:
        depth_feature_tuple = (int(depth_feature[0]), int(depth_feature[1]))

    goal_distance = env_payload.get("goal_distance_range")
    goal_distance_tuple: Optional[Tuple[float, float]] = None
    if isinstance(goal_distance, (list, tuple)) and len(goal_distance) == 2:
        goal_distance_tuple = (float(goal_distance[0]), float(goal_distance[1]))

    env = EnvConfig(
        max_speed=env_payload.get("max_speed"),
        step_duration=env_payload.get("step_duration"),
        max_episode_steps=env_payload.get("max_episode_steps"),
        success_threshold=env_payload.get("success_threshold", 0.0),
        depth_feature_size=depth_feature_tuple,
        max_depth_m=env_payload.get("max_depth_m"),
        spawn_points_json=resolve_path(config_dir, env_payload.get("spawn_points_json", ""))
        if env_payload.get("spawn_points_json")
        else None,
        goal_distance_range=goal_distance_tuple,
    )

    deterministic = bool(eval_payload.get("deterministic", True))
    if args.stochastic:
        deterministic = False
    if args.deterministic:
        deterministic = True

    eval_cfg = EvalConfig(
        num_episodes=int(
            args.num_episodes
            if args.num_episodes is not None
            else eval_payload.get("num_episodes", 0)
        ),
        model_path=resolve_path(
            config_dir,
            args.model_path if args.model_path is not None else eval_payload.get("model_path", ""),
        ),
        model_algo=str(
            args.model_algo if args.model_algo is not None else eval_payload.get("model_algo", "TD3")
        ),
        deterministic=deterministic,
        max_steps_per_episode=int(
            args.max_steps_per_episode
            if args.max_steps_per_episode is not None
            else eval_payload.get("max_steps_per_episode", 400)
        ),
        waypoint_tolerance=float(
            args.waypoint_tolerance
            if args.waypoint_tolerance is not None
            else eval_payload.get("waypoint_tolerance", 3.0)
        ),
        output_json=resolve_path(
            config_dir,
            args.output_json if args.output_json is not None else eval_payload.get("output_json", "eval_results.json"),
        ),
        log_level=str(
            args.log_level if args.log_level is not None else eval_payload.get("log_level", "INFO")
        ),
        seed=int(args.seed if args.seed is not None else eval_payload.get("seed", 42)),
        shuffle_paths=bool(args.shuffle_paths or eval_payload.get("shuffle_paths", False)),
        visualize=bool(eval_payload.get("visualize", False)),
        visualize_step_interval=int(eval_payload.get("visualize_step_interval", 1)),
        visualize_pause_sec=float(eval_payload.get("visualize_pause_sec", 0.0)),
        visualize_window_scale=float(eval_payload.get("visualize_window_scale", 1.0)),
        interactive_select=bool(args.interactive_select or eval_payload.get("interactive_select", False)),
        interactive_vertical_speed=float(eval_payload.get("interactive_vertical_speed", 1.0)),
        interactive_takeoff_clearance=float(eval_payload.get("interactive_takeoff_clearance", 0.5)),
        interactive_landing_clearance=float(eval_payload.get("interactive_landing_clearance", 1.5)),
        interactive_landing_timeout_sec=float(eval_payload.get("interactive_landing_timeout_sec", 60.0)),
    )

    if not eval_cfg.model_path:
        raise ValueError("eval.model_path is required")
    if eval_cfg.interactive_vertical_speed <= 0.0:
        raise ValueError("eval.interactive_vertical_speed must be > 0")
    if eval_cfg.interactive_takeoff_clearance < 0.0:
        raise ValueError("eval.interactive_takeoff_clearance must be >= 0")
    if eval_cfg.interactive_landing_clearance <= 0.0:
        raise ValueError("eval.interactive_landing_clearance must be > 0")
    if eval_cfg.interactive_landing_timeout_sec <= 0.0:
        raise ValueError("eval.interactive_landing_timeout_sec must be > 0")

    return EvaluationConfig(planner=planner, paths=paths, env=env, eval=eval_cfg)


def setup_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
    )
    return logging.getLogger("evaluate")


def build_env(env_cfg: EnvConfig, logger: logging.Logger) -> UAVSimpleTrainEnv:
    signature = inspect.signature(UAVSimpleTrainEnv)
    accepted = set(signature.parameters.keys())
    env_kwargs: Dict[str, Any] = {}

    if env_cfg.max_speed is not None:
        env_kwargs["max_speed"] = float(env_cfg.max_speed)
    if env_cfg.step_duration is not None:
        env_kwargs["step_duration"] = float(env_cfg.step_duration)
    if env_cfg.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = int(env_cfg.max_episode_steps)
    if env_cfg.success_threshold is not None:
        env_kwargs["success_threshold"] = float(env_cfg.success_threshold)
    if env_cfg.depth_feature_size is not None:
        env_kwargs["depth_feature_size"] = env_cfg.depth_feature_size
    if env_cfg.max_depth_m is not None:
        env_kwargs["max_depth_m"] = float(env_cfg.max_depth_m)
    if env_cfg.spawn_points_json is not None:
        env_kwargs["spawn_points_json"] = env_cfg.spawn_points_json
    if env_cfg.goal_distance_range is not None:
        env_kwargs["goal_distance_range"] = env_cfg.goal_distance_range

    filtered_kwargs = {k: v for k, v in env_kwargs.items() if k in accepted}
    ignored = sorted(set(env_kwargs.keys()) - set(filtered_kwargs.keys()))
    if ignored:
        logger.warning("Ignored unknown env kwargs: %s", ", ".join(ignored))

    return UAVSimpleTrainEnv(**filtered_kwargs)


def validate_model_env_spaces(model: BaseAlgorithm, env: UAVSimpleTrainEnv) -> None:
    model_obs_space = getattr(model, "observation_space", None)
    model_act_space = getattr(model, "action_space", None)
    env_obs_space = getattr(env, "observation_space", None)
    env_act_space = getattr(env, "action_space", None)

    if not isinstance(model_obs_space, spaces.Space) or not isinstance(
        env_obs_space, spaces.Space
    ):
        raise ValueError("Cannot read model/env observation space for compatibility check")

    if not isinstance(model_act_space, spaces.Space) or not isinstance(
        env_act_space, spaces.Space
    ):
        raise ValueError("Cannot read model/env action space for compatibility check")

    if model_obs_space.shape != env_obs_space.shape:
        raise ValueError(
            "Observation space mismatch: "
            f"model={model_obs_space.shape}, env={env_obs_space.shape}"
        )

    if model_act_space.shape != env_act_space.shape:
        raise ValueError(
            "Action space mismatch: "
            f"model={model_act_space.shape}, env={env_act_space.shape}"
        )


def parse_reset_output(reset_output: Any) -> Tuple[np.ndarray, dict]:
    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        obs, info = reset_output
        return np.asarray(obs, dtype=np.float32), dict(info)
    return np.asarray(reset_output, dtype=np.float32), {}


def parse_step_output(step_output: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
    if not isinstance(step_output, tuple):
        raise ValueError("env.step(...) must return a tuple")

    if len(step_output) == 5:
        obs, reward, terminated, truncated, info = step_output
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
        )

    if len(step_output) == 4:
        obs, reward, done, info = step_output
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), False, dict(info)

    raise ValueError(f"Unsupported env.step(...) output length: {len(step_output)}")


def get_current_position_from_env(env: UAVSimpleTrainEnv) -> np.ndarray:
    if not hasattr(env, "_get_kinematics"):
        raise RuntimeError("Environment does not expose _get_kinematics")

    kin_output = env._get_kinematics()
    if isinstance(kin_output, tuple) and len(kin_output) >= 1:
        pos = np.asarray(kin_output[0], dtype=np.float32)
        if pos.shape == (3,):
            return pos

    raise RuntimeError("Cannot parse current position from env._get_kinematics()")


def update_env_target(
    env: UAVSimpleTrainEnv,
    target: np.ndarray,
    fallback_obs: np.ndarray,
    segment_start: Optional[np.ndarray] = None,
) -> np.ndarray:
    if hasattr(env, "set_navigation_target"):
        return np.asarray(
            env.set_navigation_target(target, segment_start=segment_start),
            dtype=np.float32,
        )

    if segment_start is not None and hasattr(env, "start_pos"):
        env.start_pos = np.asarray(segment_start, dtype=np.float32).copy()

    if hasattr(env, "target_pos"):
        env.target_pos = np.asarray(target, dtype=np.float32).copy()

    if hasattr(env, "prev_distance") and hasattr(env, "_compute_distance"):
        current_pos = get_current_position_from_env(env)
        env.prev_distance = float(env._compute_distance(current_pos, env.target_pos))

    if hasattr(env, "best_distance") and hasattr(env, "prev_distance"):
        env.best_distance = float(env.prev_distance)

    if hasattr(env, "_get_obs"):
        return np.asarray(env._get_obs(), dtype=np.float32)

    return fallback_obs


def compute_waypoint_length(waypoints: Sequence[np.ndarray]) -> float:
    if len(waypoints) < 2:
        return 0.0

    total = 0.0
    for idx in range(len(waypoints) - 1):
        delta = np.asarray(waypoints[idx + 1]) - np.asarray(waypoints[idx])
        total += float(np.linalg.norm(delta))
    return total


def compute_spl(success: bool, ideal_length: float, actual_length: float) -> float:
    if not success or ideal_length <= 0.0:
        return 0.0
    denom = max(ideal_length, max(actual_length, 1e-6))
    return float(ideal_length / denom)


def load_paths_json(path: str) -> Tuple[List[PathPair], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    meta: Dict[str, Any] = {}
    paths: List[PathPair] = []

    if isinstance(payload, dict):
        meta = payload
        raw_paths = payload.get("paths", [])
    elif isinstance(payload, list):
        raw_paths = payload
    else:
        raise ValueError("paths.json must be a list or a dict with 'paths' field")

    for item in raw_paths:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        end = item.get("end")
        if not isinstance(start, dict) or not isinstance(end, dict):
            continue
        start_vec = np.array(
            [float(start["x"]), float(start["y"]), float(start["z"])], dtype=np.float32
        )
        end_vec = np.array(
            [float(end["x"]), float(end["y"]), float(end["z"])], dtype=np.float32
        )
        paths.append(
            PathPair(
                start=start_vec,
                end=end_vec,
                path_id=item.get("id"),
                distance=float(item["distance"]) if "distance" in item else None,
            )
        )

    if not paths:
        raise ValueError(f"No valid path pairs found in {path}")

    return paths, meta


def select_path_pairs(
    paths: List[PathPair],
    num_episodes: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> List[PathPair]:
    indices = list(range(len(paths)))
    if shuffle:
        rng.shuffle(indices)

    if num_episodes <= 0:
        return [paths[i] for i in indices]

    if num_episodes <= len(paths):
        return [paths[i] for i in indices[:num_episodes]]

    selected = [paths[i] for i in indices]
    while len(selected) < num_episodes:
        for idx in indices:
            if len(selected) >= num_episodes:
                break
            selected.append(paths[idx])
    return selected


def episode_to_dict(result: EpisodeResult) -> Dict[str, Any]:
    return {
        "episode_id": result.episode_id,
        "path_id": result.path_id,
        "success": result.success,
        "collision": result.collision,
        "timeout": result.timeout,
        "planning_failed": result.planning_failed,
        "failed": result.failed,
        "steps": result.steps,
        "duration_sec": result.duration_sec,
        "ideal_path_length": result.ideal_path_length,
        "actual_path_length": result.actual_path_length,
        "spl": result.spl,
        "waypoint_count": result.waypoint_count,
        "completed_waypoints": result.completed_waypoints,
        "start": result.start,
        "end": result.end,
        "final_position": result.final_position,
        "final_distance_to_goal": result.final_distance_to_goal,
        "reason": result.reason,
    }


def resolve_height(paths_cfg: PathsConfig, meta: Dict[str, Any], path_pair: PathPair) -> float:
    if paths_cfg.height is not None:
        return float(paths_cfg.height)
    if isinstance(meta, dict) and "height" in meta and meta["height"] is not None:
        return float(meta["height"])
    return float(path_pair.start[2])


def map_surface_z_at_position(
    planner: AStarPlanner,
    mapper: PixelToAirSimMatrixMapper,
    position: np.ndarray,
) -> Tuple[float, Pixel]:
    """Estimate a physical landing surface from the calibrated map height channel."""
    u, v = mapper.airsim_xy_to_pixel((float(position[0]), float(position[1])))
    pixel = (int(round(u)), int(round(v)))
    if not planner.in_bounds(pixel):
        raise ValueError(f"Interactive surface pixel {pixel} is outside map bounds")
    local_height = max(0.0, float(planner.height_map[pixel[1], pixel[0]]))
    return -local_height, pixel


def build_interactive_flight_profile(
    planner: AStarPlanner,
    mapper: PixelToAirSimMatrixMapper,
    path_pair: PathPair,
    eval_cfg: EvalConfig,
    logger: logging.Logger,
) -> InteractiveFlightProfile:
    start_surface_z, start_pixel = map_surface_z_at_position(
        planner, mapper, path_pair.start
    )
    landing_surface_z, landing_pixel = map_surface_z_at_position(
        planner, mapper, path_pair.end
    )
    launch_position = np.array(
        [
            float(path_pair.start[0]),
            float(path_pair.start[1]),
            start_surface_z - eval_cfg.interactive_takeoff_clearance,
        ],
        dtype=np.float32,
    )
    landing_position = np.array(
        [
            float(path_pair.end[0]),
            float(path_pair.end[1]),
            landing_surface_z,
        ],
        dtype=np.float32,
    )
    if float(path_pair.start[2]) >= float(launch_position[2]):
        raise ValueError(
            "Interactive cruise altitude must be above the selected takeoff surface: "
            f"cruise_z={path_pair.start[2]:.2f}, surface_z={start_surface_z:.2f}"
        )
    approach_z = landing_surface_z - eval_cfg.interactive_landing_clearance
    if float(path_pair.end[2]) >= approach_z:
        raise ValueError(
            "Interactive cruise altitude must be above the landing approach height: "
            f"cruise_z={path_pair.end[2]:.2f}, approach_z={approach_z:.2f}"
        )

    logger.info(
        "Interactive surfaces: takeoff_pixel=%s surface_z=%.2f, "
        "landing_pixel=%s surface_z=%.2f, landing_approach_z=%.2f",
        start_pixel,
        start_surface_z,
        landing_pixel,
        landing_surface_z,
        approach_z,
    )
    return InteractiveFlightProfile(
        launch_position=launch_position,
        landing_position=landing_position,
        launch_surface_z=start_surface_z,
        vertical_speed=eval_cfg.interactive_vertical_speed,
        landing_approach_clearance=eval_cfg.interactive_landing_clearance,
        landing_timeout_sec=eval_cfg.interactive_landing_timeout_sec,
    )


def resolve_model_path(path: str) -> str:
    if not path:
        return path
    if path.endswith(".zip"):
        return path
    candidate = f"{path}.zip"
    if os.path.exists(candidate):
        return candidate
    return path


def main() -> None:
    args = parse_args()
    config = load_config(args)

    logger = setup_logging(config.eval.log_level)

    planner_cfg = config.planner
    paths_cfg = config.paths

    if not os.path.exists(planner_cfg.map_path):
        raise FileNotFoundError(f"Map file not found: {planner_cfg.map_path}")
    if not os.path.exists(planner_cfg.pixel_to_airsim_matrix):
        raise FileNotFoundError(
            f"Matrix file not found: {planner_cfg.pixel_to_airsim_matrix}"
        )
    if not os.path.exists(paths_cfg.paths_json):
        raise FileNotFoundError(f"Paths JSON not found: {paths_cfg.paths_json}")

    eval_cfg = config.eval
    eval_cfg.model_path = resolve_model_path(eval_cfg.model_path)
    if not os.path.exists(eval_cfg.model_path):
        raise FileNotFoundError(f"Model zip not found: {eval_cfg.model_path}")

    mapper = PixelToAirSimMatrixMapper.from_file(planner_cfg.pixel_to_airsim_matrix)
    if planner_cfg.planning_mode == "3d":
        a_star: AStarPlanner = AStarPlanner3D(
            planner_cfg.map_path,
            obstacle_height_threshold=planner_cfg.obstacle_height_threshold,
            height_channel=planner_cfg.height_channel,
            height_weight=planner_cfg.height_weight,
        )
    else:
        a_star = AStarPlanner(
            planner_cfg.map_path,
            obstacle_height_threshold=planner_cfg.obstacle_height_threshold,
            height_channel=planner_cfg.height_channel,
        )

    visualizer: Optional[MapVisualizer] = None
    if eval_cfg.visualize or eval_cfg.interactive_select:
        try:
            visualizer = MapVisualizer(
                map_path=planner_cfg.map_path,
                mapper=mapper,
                step_interval=eval_cfg.visualize_step_interval,
                pause_sec=eval_cfg.visualize_pause_sec,
                window_scale=eval_cfg.visualize_window_scale,
                logger=logger,
            )
            visualizer.enable_log_mirroring()
        except Exception as exc:
            logger.warning("Visualization disabled: %s", exc)
            visualizer = None

    if eval_cfg.interactive_select:
        if visualizer is None:
            raise RuntimeError("Interactive selection requires visualization")
        z_up = float(paths_cfg.height) if paths_cfg.height is not None else -5.0
        selection = visualizer.select_start_goal(z_up)
        if selection is None:
            logger.info("Interactive selection cancelled.")
            if visualizer is not None:
                visualizer.close()
            return
        start, end = selection
        path_pairs = [PathPair(start=start, end=end, path_id=None, distance=None)]
        selected_paths = path_pairs
        meta: Dict[str, Any] = {}
    else:
        logger.info("Loading paths: %s", paths_cfg.paths_json)
        path_pairs, meta = load_paths_json(paths_cfg.paths_json)

        rng = np.random.default_rng(eval_cfg.seed)
        selected_paths = select_path_pairs(
            path_pairs, eval_cfg.num_episodes, rng, eval_cfg.shuffle_paths
        )

    env = build_env(config.env, logger)

    if config.env.spawn_points_json:
        logger.warning(
            "spawn_points_json is set but will be ignored for path-based evaluation"
        )
        if hasattr(env, "_use_spawn_points"):
            env._use_spawn_points = False
    navigator = SB3LocalNavigator(
        model_path=eval_cfg.model_path,
        algo=eval_cfg.model_algo,
        deterministic=eval_cfg.deterministic,
    )
    validate_model_env_spaces(navigator.model, env)

    if config.env.max_episode_steps is None:
        env.max_episode_steps = int(eval_cfg.max_steps_per_episode)

    if config.env.success_threshold is None:
        env.success_threshold = 0.0

    executor = EpisodeExecutor(
        env=env,
        navigator=navigator,
        waypoint_tolerance=eval_cfg.waypoint_tolerance,
        max_steps=eval_cfg.max_steps_per_episode,
        logger=logger,
        visualizer=visualizer,
    )

    metrics = MetricsTracker()

    logger.info(
        "Starting evaluation: %d episodes, planning_mode=%s",
        len(selected_paths),
        planner_cfg.planning_mode,
    )
    for idx, path_pair in enumerate(selected_paths, start=1):
        z_up = resolve_height(paths_cfg, meta, path_pair)
        planner = GlobalPlanner(
            planner=a_star,
            mapper=mapper,
            waypoint_stride=planner_cfg.waypoint_stride,
            z_up=z_up,
            planning_mode=planner_cfg.planning_mode,
            height_lift_coef=planner_cfg.height_lift_coef,
            max_segment_altitude_delta=planner_cfg.max_segment_altitude_delta,
        )

        try:
            plan = planner.plan(path_pair.start, path_pair.end)
            interactive_profile = (
                build_interactive_flight_profile(
                    a_star, mapper, path_pair, eval_cfg, logger
                )
                if eval_cfg.interactive_select
                else None
            )
        except Exception as exc:
            logger.error(
                "Episode %d planning failed (path_id=%s): %s",
                idx,
                path_pair.path_id,
                exc,
            )
            result = EpisodeResult(
                episode_id=idx,
                path_id=path_pair.path_id,
                success=False,
                collision=False,
                timeout=True,
                planning_failed=True,
                failed=True,
                steps=0,
                duration_sec=0.0,
                ideal_path_length=0.0,
                actual_path_length=0.0,
                spl=0.0,
                waypoint_count=0,
                completed_waypoints=0,
                start=path_pair.start.tolist(),
                end=path_pair.end.tolist(),
                final_position=None,
                final_distance_to_goal=None,
                reason="planning_failed",
            )
            metrics.add(result)
            continue

        z_values = [float(waypoint[2]) for waypoint in plan.waypoints]
        raw_z_range = plan.raw_altitude_range or (min(z_values), max(z_values))
        logger.info(
            "Episode %d: path_id=%s, mode=%s, waypoints=%d, "
            "ideal_len=%.2f, raw_z_range=[%.2f, %.2f], "
            "execution_z_range=[%.2f, %.2f], transition_waypoints=%d",
            idx,
            path_pair.path_id,
            planner_cfg.planning_mode,
            len(plan.waypoints),
            plan.ideal_length,
            raw_z_range[0],
            raw_z_range[1],
            min(z_values),
            max(z_values),
            plan.inserted_transition_waypoint_count,
        )

        result = executor.run_episode(
            idx, path_pair, plan, interactive_profile=interactive_profile
        )
        metrics.add(result)

        logger.info(
            "Episode %d result: success=%s collision=%s timeout=%s steps=%d spl=%.3f",
            idx,
            result.success,
            result.collision,
            result.timeout,
            result.steps,
            result.spl,
        )

    summary = metrics.summary()
    logger.info(
        "Summary: SR=%.2f CR=%.2f TR=%.2f SPL=%.3f avg_steps=%.1f avg_time=%.2f failed=%d avg_fail_dist=%.2f",
        summary["success_rate"],
        summary["collision_rate"],
        summary["timeout_rate"],
        summary["spl"],
        summary["avg_steps_success"],
        summary["avg_time_sec_success"],
        summary["failed_count"],
        summary["avg_failed_distance"],
    )

    if eval_cfg.output_json:
        output_dir = os.path.dirname(eval_cfg.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(eval_cfg.output_json, "w", encoding="utf-8") as handle:
            json.dump(metrics.to_json(), handle, indent=2)
        logger.info("Saved results to %s", eval_cfg.output_json)

    env.close()
    if visualizer is not None:
        if eval_cfg.interactive_select and not visualizer.closed:
            logger.info(
                "Interactive execution finished. Close the visualization window to exit."
            )
            visualizer.wait_until_closed()
        visualizer.close()


if __name__ == "__main__":
    main()
