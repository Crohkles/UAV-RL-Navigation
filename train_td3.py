import argparse
import os
import re
from datetime import datetime

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.noise import NormalActionNoise

from envs import UAVSimpleTrainEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TD3 UAV training entry")
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
        default="td3_uav_simple",
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
    """从文件名中提取历史步数，如 xxx_2000_steps.zip。"""
    file_name = os.path.basename(path)
    match = re.search(r"_(\d+)_steps(?:\.zip)?$", file_name)
    if not match:
        return 0
    return int(match.group(1))


def _derive_replay_buffer_path_from_checkpoint(model_zip_path: str) -> str:
    """按 SB3 CheckpointCallback 规则推导 replay buffer 路径。"""
    file_name = os.path.basename(model_zip_path)
    # SB3 命名规则: <name_prefix>_<steps>_steps.zip -> <name_prefix>_replay_buffer_<steps>_steps.pkl
    match = re.search(r"^(?P<prefix>.+)_(?P<steps>\d+)_steps\.zip$", file_name)
    if not match:
        return ""

    replay_file_name = (
        f"{match.group('prefix')}_replay_buffer_{match.group('steps')}_steps.pkl"
    )
    return os.path.join(os.path.dirname(model_zip_path), replay_file_name)


def _sanitize_run_name(raw: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw.strip())
    return safe.strip("_")


class LossTensorboardCallback(BaseCallback):
    """将 actor/critic loss 显式写入 TensorBoard，便于稳定可视化。"""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._tb_writer = None
        self._last_logged_step = -1
        self._last_actor_loss = None
        self._last_critic_loss = None

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, (float, int, np.floating, np.integer)):
            return float(value)
        return None

    def _on_training_start(self) -> None:
        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                self._tb_writer = fmt.writer
                break

        if self._tb_writer is None and self.verbose > 0:
            print("[WARN] TensorBoard writer not found; actor/critic loss callback disabled.")

    def _on_step(self) -> bool:
        if self._tb_writer is None:
            return True

        metrics = getattr(self.model.logger, "name_to_value", {})
        if not isinstance(metrics, dict):
            return True

        actor_loss = self._to_scalar(metrics.get("train/actor_loss"))
        critic_loss = self._to_scalar(metrics.get("train/critic_loss"))

        if actor_loss is None and critic_loss is None:
            return True

        current_step = int(getattr(self.model, "num_timesteps", self.num_timesteps))
        if (
            current_step == self._last_logged_step
            and actor_loss == self._last_actor_loss
            and critic_loss == self._last_critic_loss
        ):
            return True

        if actor_loss is not None:
            self._tb_writer.add_scalar("train/actor_loss", actor_loss, current_step)
        if critic_loss is not None:
            self._tb_writer.add_scalar("train/critic_loss", critic_loss, current_step)

        self._last_logged_step = current_step
        self._last_actor_loss = actor_loss
        self._last_critic_loss = critic_loss
        return True


def main() -> None:
    """
    TD3 极简训练入口。

    关键点：
    1) 使用 MlpPolicy，对应一维向量观测输入。
    2) 配置高斯动作噪声，提升连续动作空间探索能力。
    3) 打开 TensorBoard 日志，支持自动 checkpoint 与断点续训。
    4) 续训时尽量继承历史步数，并为每次训练生成独立批次目录避免覆盖。
    """
    args = _parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(project_root, "logs", "tensorboard")
    model_dir = os.path.join(project_root, "models")
    checkpoint_root = os.path.join(model_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint_root, exist_ok=True)

    env = UAVSimpleTrainEnv()

    n_actions = env.action_space.shape[-1]
    # TD3 常见做法：为每个动作维度设置同尺度高斯噪声。
    # 这里 sigma=0.2，表示在 [-1,1] 动作系数空间内提供适中的探索扰动。
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
            env=env,
            verbose=1,
            tensorboard_log=log_dir,
        )
        # 继续训练时显式恢复动作噪声，保证探索策略一致。
        model.action_noise = action_noise

        # 继承历史训练步数：优先模型内记录，缺失时从checkpoint文件名兜底。
        loaded_steps = int(getattr(model, "num_timesteps", 0))
        hinted_steps = _extract_steps_from_checkpoint_name(resume_zip)
        resume_base_steps = max(loaded_steps, hinted_steps)
        model.num_timesteps = resume_base_steps

        # 若存在配套回放缓存则一并加载，可提升续训稳定性。
        replay_path = _derive_replay_buffer_path_from_checkpoint(resume_zip)
        if replay_path and os.path.exists(replay_path):
            model.load_replay_buffer(replay_path)
            print(f"已加载 replay buffer: {replay_path}")

        print(f"继续训练模型: {resume_zip} | 历史步数: {resume_base_steps}")
    else:
        model = TD3(
            policy="MlpPolicy",
            env=env,
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
    loss_callback = LossTensorboardCallback()
    callbacks = CallbackList([checkpoint_callback, loss_callback])

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=callbacks,
        reset_num_timesteps=not resume_mode,
    )

    # 保存两份：归档模型防覆盖 + 固定别名方便下次快速续训。
    archived_model_path = os.path.join(model_dir, f"{args.save_name}_{run_name}")
    latest_model_path = os.path.join(model_dir, args.save_name)
    model.save(archived_model_path)
    model.save(latest_model_path)
    env.close()

    print(f"训练完成，归档模型: {archived_model_path}.zip")
    print(f"训练完成，最新模型别名: {latest_model_path}.zip")
    print(f"自动 checkpoint 目录: {checkpoint_dir}")
    print(f"checkpoint 保存频率: 每 {args.checkpoint_freq} timesteps")
    print(f"TensorBoard 日志目录: {log_dir}")


if __name__ == "__main__":
    main()
