import argparse
import os
from typing import List

import numpy as np
from stable_baselines3 import TD3

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
    return parser.parse_args()


def resolve_model_zip_path(model_path: str) -> str:
    if model_path.endswith(".zip"):
        return model_path
    return f"{model_path}.zip"


def is_target_unique(candidate: np.ndarray, used_targets: List[np.ndarray]) -> bool:
    for old in used_targets:
        if float(np.linalg.norm(candidate - old)) < 0.5:
            return False
    return True


def sample_unique_target(
    rng: np.random.Generator,
    target_min: np.ndarray,
    target_max: np.ndarray,
    used_targets: List[np.ndarray],
) -> np.ndarray:
    while True:
        target = rng.uniform(target_min, target_max).astype(np.float32)
        if is_target_unique(target, used_targets):
            return target


def run_one_episode(
    env: UAVSimpleTrainEnv,
    model: TD3,
    target: np.ndarray,
    deterministic: bool,
    max_steps: int,
) -> tuple[bool, dict]:
    # Reset to the same start state as training.
    env.reset()

    # Keep the target fixed for retries of the same test case.
    env.target_pos = target.astype(np.float32)
    current_pos, _ = env._get_kinematics()
    env.prev_distance = env._compute_distance(current_pos, env.target_pos)
    env.current_step = 0

    obs = env._get_obs()
    last_info: dict = {
        "distance_to_target": env.prev_distance,
        "collision": False,
    }

    terminated = False
    truncated = False

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, info = env.step(action)
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

    env = UAVSimpleTrainEnv()
    model = TD3.load(model_zip)

    max_steps = (
        env.max_episode_steps if args.max_steps_per_episode <= 0 else args.max_steps_per_episode
    )
    deterministic = not args.stochastic

    rng = np.random.default_rng(args.seed)
    successful_targets: List[np.ndarray] = []

    print("=== AirSim policy test start ===")
    print(f"Model: {model_zip}")
    print(f"Start position (NED): {env.start_pos.tolist()}")
    print(f"Target range min: {env.target_min.tolist()}")
    print(f"Target range max: {env.target_max.tolist()}")
    print(f"Need successful targets: {args.num_targets}")

    target_index = 0
    total_attempts = 0

    while target_index < args.num_targets:
        target = sample_unique_target(rng, env.target_min, env.target_max, successful_targets)
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
                print(
                    f"SUCCESS on attempt {attempt} | distance={distance:.3f}m | position={pos_text}"
                )
            else:
                fail_reason = "collision" if collision else "not_reached_or_timeout"
                print(
                    f"FAIL on attempt {attempt} | reason={fail_reason} "
                    f"| distance={distance:.3f}m | position={pos_text}"
                )

                if args.max_attempts_per_target > 0 and attempt >= args.max_attempts_per_target:
                    raise RuntimeError(
                        "Reached max attempts for one target without success. "
                        "Increase --max-attempts-per-target or use a better checkpoint."
                    )

    env.close()

    print("\n========================================")
    print("Test finished.")
    print(f"Successful targets: {len(successful_targets)} / {args.num_targets}")
    print(f"Total attempts: {total_attempts}")


if __name__ == "__main__":
    main()
