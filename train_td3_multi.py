import argparse
import os
import re
from datetime import datetime

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.uav_multi_env import UAVMultiTrainEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TD3 multi-drone parallel training")
    parser.add_argument(
        "--num-drones",
        type=int,
        default=4,
        help="Number of parallel drones in AirSim.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000,
        help="Total training timesteps for this run.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=1_000,
        help="Save checkpoint every N timesteps.",
    )
    parser.add_argument(
        "--resume-model",
        type=str,
        default="",
        help="Path to an existing TD3 model (.zip or path without suffix).",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="td3_uav_multi",
        help="Base name for final model and checkpoints.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run name. If empty, script auto-generates one.",
    )
    return parser.parse_args()


def _resolve_model_zip_path(path: str) -> str:
    if path.endswith(".zip"):
        return path
    return f"{path}.zip"


def _extract_steps_from_checkpoint_name(path: str) -> int:
    file_name = os.path.basename(path)
    match = re.search(r"_(\d+)_steps(?:\.zip)?$", file_name)
    if not match:
        return 0
    return int(match.group(1))


def _sanitize_run_name(raw: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw.strip())
    return safe.strip("_")


def _make_env(vehicle_name: str, start_pos: np.ndarray, seed: int):
    """创建单个环境的工厂函数，供 SubprocVecEnv 使用。"""

    def _init() -> UAVMultiTrainEnv:
        env = UAVMultiTrainEnv(
            vehicle_name=vehicle_name,
            start_pos=start_pos,
        )
        env.reset(seed=seed)
        return env

    return _init


def _build_start_positions(num_drones: int) -> list[np.ndarray]:
    """为每架无人机分配不同的起始位置，y 轴间隔 10m 避免重叠碰撞。"""
    positions = []
    for i in range(num_drones):
        y_offset = (i - (num_drones - 1) / 2) * 10.0
        positions.append(np.array([0.0, y_offset, -30.0], dtype=np.float32))
    return positions


def main() -> None:
    args = _parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(project_root, "logs", "tensorboard")
    model_dir = os.path.join(project_root, "models")
    checkpoint_root = os.path.join(model_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint_root, exist_ok=True)

    start_positions = _build_start_positions(args.num_drones)
    env_fns = [
        _make_env(f"Drone{i + 1}", start_positions[i], seed=i)
        for i in range(args.num_drones)
    ]
    vec_env = SubprocVecEnv(env_fns)
    print(f"已创建 {args.num_drones} 个并行环境 (SubprocVecEnv)")

    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions, dtype=np.float32),
        sigma=0.2 * np.ones(n_actions, dtype=np.float32),
    )

    resume_mode = bool(args.resume_model)
    resume_base_steps = 0

    if resume_mode:
        resume_zip = _resolve_model_zip_path(args.resume_model)
        if not os.path.exists(resume_zip):
            raise FileNotFoundError(f"Resume model not found: {resume_zip}")

        model = TD3.load(
            resume_zip,
            env=vec_env,
            verbose=1,
            tensorboard_log=log_dir,
        )
        model.action_noise = action_noise

        loaded_steps = int(getattr(model, "num_timesteps", 0))
        hinted_steps = _extract_steps_from_checkpoint_name(resume_zip)
        resume_base_steps = max(loaded_steps, hinted_steps)
        model.num_timesteps = resume_base_steps

        replay_path = resume_zip.replace(".zip", "_replay_buffer.pkl")
        if os.path.exists(replay_path):
            model.load_replay_buffer(replay_path)
            print(f"已加载 replay buffer: {replay_path}")

        print(f"继续训练模型: {resume_zip} | 历史步数: {resume_base_steps}")
    else:
        model = TD3(
            policy="MlpPolicy",
            env=vec_env,
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=log_dir,
        )
        print("从头开始训练新模型")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_name = _sanitize_run_name(args.run_name)
        if not run_name:
            run_name = f"{args.save_name}_{stamp}"
    elif resume_mode:
        run_name = f"{args.save_name}_resume_from_{resume_base_steps}_{stamp}"
    else:
        run_name = f"{args.save_name}_fresh_{stamp}"

    checkpoint_dir = os.path.join(checkpoint_root, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix=f"{args.save_name}_ckpt",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=checkpoint_callback,
        reset_num_timesteps=not resume_mode,
    )

    archived_model_path = os.path.join(model_dir, f"{args.save_name}_{run_name}")
    latest_model_path = os.path.join(model_dir, args.save_name)
    model.save(archived_model_path)
    model.save(latest_model_path)
    vec_env.close()

    print(f"训练完成，归档模型: {archived_model_path}.zip")
    print(f"训练完成，最新模型别名: {latest_model_path}.zip")
    print(f"自动 checkpoint 目录: {checkpoint_dir}")
    print(f"checkpoint 保存频率: 每 {args.checkpoint_freq} timesteps")
    print(f"TensorBoard 日志目录: {log_dir}")


if __name__ == "__main__":
    main()
