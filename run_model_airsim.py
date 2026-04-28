import argparse
import inspect
import json
import os
import threading
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from envs import UAVSimpleTrainEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained TD3 model in AirSim for multiple random targets."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model zip, or path without .zip suffix.",
    )
    parser.add_argument(
        "--num-targets",
        type=int,
        default=3,
        help="How many different random targets to test.",
    )
    parser.add_argument(
        "--max-attempts-per-target",
        type=int,
        default=0,
        help="0 means unlimited retries until success for that target.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=0,
        help="0 means use env.max_episode_steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible target sampling.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy inference (default is deterministic).",
    )
    parser.add_argument(
        "--env-kwargs-json",
        type=str,
        default="",
        help=(
            "Optional JSON string to override env kwargs, e.g. "
            "'{\"max_episode_steps\":300,\"yaw_rate_penalty_weight\":0.02}'"
        ),
    )
    parser.add_argument(
        "--show-reward-breakdown",
        action="store_true",
        help="Print reward/penalty/bonus terms from env info when available.",
    )
    parser.add_argument(
        "--close-timeout-sec",
        type=float,
        default=3.0,
        help=(
            "Timeout for graceful env.close(). If exceeded, fallback to fast control release "
            "to avoid long blocking at program end."
        ),
    )
    return parser.parse_args()


def resolve_model_zip_path(model_path: str) -> str:
    if model_path.endswith(".zip"):
        return model_path
    return f"{model_path}.zip"


def parse_env_kwargs_json(raw_json: str) -> Dict[str, Any]:
    if not raw_json.strip():
        return {}

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --env-kwargs-json: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("--env-kwargs-json must decode to a JSON object")
    return parsed


def build_env(env_kwargs: Dict[str, Any]) -> UAVSimpleTrainEnv:
    # 过滤未知参数，降低脚本与环境构造函数耦合。
    signature = inspect.signature(UAVSimpleTrainEnv)
    accepted = set(signature.parameters.keys())
    filtered_kwargs: Dict[str, Any] = {}
    ignored_kwargs: List[str] = []

    for key, value in env_kwargs.items():
        if key in accepted:
            filtered_kwargs[key] = value
        else:
            ignored_kwargs.append(key)

    if ignored_kwargs:
        ignored_text = ", ".join(sorted(ignored_kwargs))
        print(f"[WARN] Ignored unknown env kwargs: {ignored_text}")

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
            f"model={model_obs_space.shape}, env={env_obs_space.shape}. "
            "Please use a checkpoint trained with current env observation design."
        )

    if model_act_space.shape != env_act_space.shape:
        raise ValueError(
            "Action space mismatch: "
            f"model={model_act_space.shape}, env={env_act_space.shape}. "
            "Please use a checkpoint trained with current env action design."
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

    raise RuntimeError("Cannot parse current position from env._get_kinematics() output")


def force_episode_target(env: UAVSimpleTrainEnv, target: np.ndarray, reset_obs: np.ndarray) -> np.ndarray:
    if not hasattr(env, "target_pos"):
        raise RuntimeError("Environment does not expose target_pos")

    env.target_pos = target.astype(np.float32)

    if hasattr(env, "current_step"):
        env.current_step = 0

    if hasattr(env, "prev_action"):
        env.prev_action = np.zeros_like(np.asarray(env.prev_action, dtype=np.float32))

    if hasattr(env, "prev_distance") and hasattr(env, "_compute_distance"):
        current_pos = get_current_position_from_env(env)
        env.prev_distance = float(env._compute_distance(current_pos, env.target_pos))

    if hasattr(env, "_get_obs"):
        return np.asarray(env._get_obs(), dtype=np.float32)

    # 若环境不暴露 _get_obs，就退回 reset 输出，保证至少可运行。
    return reset_obs


def extract_reward_breakdown(info: dict) -> Dict[str, float]:
    terms: Dict[str, float] = {}
    for key, value in info.items():
        key_lower = str(key).lower()
        if not any(token in key_lower for token in ("reward", "penalty", "bonus")):
            continue

        if isinstance(value, (int, float, np.floating, np.integer)):
            terms[str(key)] = float(value)
    return terms


def format_reward_breakdown(info: dict) -> str:
    terms = extract_reward_breakdown(info)
    if not terms:
        return ""
    parts = [f"{k}={v:.4f}" for k, v in sorted(terms.items())]
    return " | ".join(parts)


def fast_release_env_control(env: UAVSimpleTrainEnv) -> None:
    """快速释放控制权，避免 close 中 land join 长时间阻塞。"""
    client = getattr(env, "client", None)
    if client is None:
        return

    try:
        client.armDisarm(False)
    except Exception:
        pass

    try:
        client.enableApiControl(False)
    except Exception:
        pass


def close_env_with_timeout(env: UAVSimpleTrainEnv, timeout_sec: float) -> bool:
    """
    尝试优雅关闭环境；若超时则快速释放控制权。
    返回值：True 表示优雅关闭完成；False 表示发生超时并降级。
    """
    if timeout_sec <= 0:
        env.close()
        return True

    close_error: Dict[str, Exception] = {}

    def _close_worker() -> None:
        try:
            env.close()
        except Exception as exc:
            close_error["exc"] = exc

    thread = threading.Thread(target=_close_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        fast_release_env_control(env)
        return False

    if "exc" in close_error:
        raise close_error["exc"]

    return True


def is_target_unique(candidate: np.ndarray, used_targets: List[np.ndarray]) -> bool:
    for old in used_targets:
        if float(np.linalg.norm(candidate - old)) < 0.5:
            return False
    return True


def normalize_target_bounds(
    target_min: np.ndarray,
    target_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    target_min_np = np.asarray(target_min, dtype=np.float32)
    target_max_np = np.asarray(target_max, dtype=np.float32)

    if target_min_np.shape != target_max_np.shape:
        raise ValueError(
            "Target bounds shape mismatch: "
            f"min={target_min_np.shape}, max={target_max_np.shape}"
        )

    sample_low = np.minimum(target_min_np, target_max_np)
    sample_high = np.maximum(target_min_np, target_max_np)
    has_reversed_axis = bool(np.any(target_min_np > target_max_np))
    return sample_low, sample_high, has_reversed_axis


def sample_unique_target(
    rng: np.random.Generator,
    sample_low: np.ndarray,
    sample_high: np.ndarray,
    used_targets: List[np.ndarray],
) -> np.ndarray:
    while True:
        target = rng.uniform(sample_low, sample_high).astype(np.float32)
        if is_target_unique(target, used_targets):
            return target
    # return np.array([20.0,0.0,-50.0],dtype=np.float32)

def run_one_episode(
    env: UAVSimpleTrainEnv,
    model: BaseAlgorithm,
    target: np.ndarray,
    deterministic: bool,
    max_steps: int,
) -> tuple[bool, dict]:
    # Reset to the same start state as training.
    reset_obs, _reset_info = parse_reset_output(env.reset())

    # Keep the target fixed for retries of the same test case.
    obs = force_episode_target(env=env, target=target, reset_obs=reset_obs)

    initial_distance = float(getattr(env, "prev_distance", 1e9))
    last_info: dict = {
        "distance_to_target": initial_distance,
        "collision": False,
    }

    terminated = False
    truncated = False

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, info = parse_step_output(env.step(action))
        last_info = info
        if terminated or truncated:
            break

    distance = float(last_info.get("distance_to_target", 1e9))
    collision = bool(last_info.get("collision", False))
    success = terminated and (not collision) and (distance < env.success_threshold)
    return success, last_info


def main() -> None:
    args = parse_args()

    model_zip = resolve_model_zip_path(args.model_path)
    if not os.path.exists(model_zip):
        raise FileNotFoundError(f"Model not found: {model_zip}")

    env_kwargs = parse_env_kwargs_json(args.env_kwargs_json)
    env = build_env(env_kwargs)
    model = TD3.load(model_zip)
    validate_model_env_spaces(model, env)

    max_steps = (
        env.max_episode_steps if args.max_steps_per_episode <= 0 else args.max_steps_per_episode
    )
    deterministic = not args.stochastic

    rng = np.random.default_rng(args.seed)
    successful_targets: List[np.ndarray] = []
    sample_low, sample_high, has_reversed_axis = normalize_target_bounds(
        env.target_min,
        env.target_max,
    )

    print("=== AirSim policy test start ===")
    print(f"Model: {model_zip}")
    print(f"Start position (NED): {env.start_pos.tolist()}")
    print(f"Target range min: {env.target_min.tolist()}")
    print(f"Target range max: {env.target_max.tolist()}")
    if has_reversed_axis:
        print(
            "[WARN] Detected reversed target bounds on at least one axis. "
            "Sampling uses per-axis sorted bounds."
        )
        print(f"Sampling range low: {sample_low.tolist()}")
        print(f"Sampling range high: {sample_high.tolist()}")
    print(f"Need successful targets: {args.num_targets}")
    print(f"Obs space shape: {env.observation_space.shape}")
    print(f"Act space shape: {env.action_space.shape}")
    if env_kwargs:
        print(f"Env kwargs overrides: {env_kwargs}")

    target_index = 0
    total_attempts = 0

    while target_index < args.num_targets:
        target = sample_unique_target(rng, sample_low, sample_high, successful_targets)
        print("\n----------------------------------------")
        print(
            f"Target #{target_index + 1} sampled (NED): "
            f"[{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]"
        )

        attempt = 0
        target_success = False

        while not target_success:
            attempt += 1
            total_attempts += 1

            success, info = run_one_episode(
                env=env,
                model=model,
                target=target,
                deterministic=deterministic,
                max_steps=max_steps,
            )

            distance = float(info.get("distance_to_target", 1e9))
            collision = bool(info.get("collision", False))
            pos = info.get("position")
            if isinstance(pos, np.ndarray) and pos.shape == (3,):
                pos_text = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
            else:
                pos_text = "N/A"

            if success:
                target_success = True
                successful_targets.append(target.copy())
                target_index += 1
                reward_text = (
                    format_reward_breakdown(info) if args.show_reward_breakdown else ""
                )
                print(
                    f"SUCCESS on attempt {attempt} | distance={distance:.3f}m | position={pos_text}"
                )
                if reward_text:
                    print(f"  Reward terms: {reward_text}")
            else:
                fail_reason = "collision" if collision else "not_reached_or_timeout"
                reward_text = (
                    format_reward_breakdown(info) if args.show_reward_breakdown else ""
                )
                print(
                    f"FAIL on attempt {attempt} | reason={fail_reason} "
                    f"| distance={distance:.3f}m | position={pos_text}"
                )
                if reward_text:
                    print(f"  Reward terms: {reward_text}")

                if args.max_attempts_per_target > 0 and attempt >= args.max_attempts_per_target:
                    raise RuntimeError(
                        "Reached max attempts for one target without success. "
                        "Increase --max-attempts-per-target or use a better checkpoint."
                    )

    print("\n========================================")
    print("Test finished.")
    print(f"Successful targets: {len(successful_targets)} / {args.num_targets}")
    print(f"Total attempts: {total_attempts}")

    shutdown_start = time.perf_counter()
    graceful_closed = close_env_with_timeout(env, timeout_sec=args.close_timeout_sec)
    shutdown_cost = time.perf_counter() - shutdown_start

    if graceful_closed:
        print(f"Shutdown mode: graceful close completed in {shutdown_cost:.2f}s")
    else:
        print(
            "Shutdown mode: graceful close timeout, switched to fast release "
            f"in {shutdown_cost:.2f}s"
        )


if __name__ == "__main__":
    main()
