import time
from typing import Optional

import airsim
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils.image_processing import get_depth_feature_for_vehicle


class UAVMultiTrainEnv(gym.Env):
    """用于多无人机并行训练的 Gymnasium 环境。

    每个实例控制一架指定的无人机，通过 vehicle_name 参数区分。
    用于 SubprocVecEnv 并行采样。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        vehicle_name: str = "Drone1",
        start_pos: Optional[np.ndarray] = None,
        max_speed: float = 5.0,
        step_duration: float = 0.5,
        max_episode_steps: int = 200,
        success_threshold: float = 2.0,
        depth_feature_size: tuple[int, int] = (12, 12),
        max_depth_m: float = 50.0,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()

        self.vehicle_name = vehicle_name
        if start_pos is not None:
            self.start_pos = np.array(start_pos, dtype=np.float32)
        else:
            self.start_pos = np.array([0.0, 0.0, -30.0], dtype=np.float32)

        self.max_speed = float(max_speed)
        self.step_duration = float(step_duration)
        self.max_episode_steps = int(max_episode_steps)
        self.success_threshold = float(success_threshold)
        self.depth_feature_size = depth_feature_size
        self.max_depth_m = float(max_depth_m)
        self.alpha = float(alpha)

        self.target_min = np.array([10.0, -15.0, -40.0], dtype=np.float32)
        self.target_max = np.array([30.0, 15.0, -20.0], dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        depth_dim = self.depth_feature_size[0] * self.depth_feature_size[1]
        obs_dim = depth_dim + 3 + 3
        self.observation_space = spaces.Box(
            low=-1_000.0,
            high=1_000.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.client = airsim.MultirotorClient()
        self._connect_client()

        self.target_pos = np.zeros(3, dtype=np.float32)
        self.prev_distance = 0.0
        self.current_step = 0
        self.v_cmd_prev = np.zeros(3, dtype=np.float32)

    def _connect_client(self) -> None:
        """建立连接并获取指定无人机的控制权。"""
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

    def _safe_reset_vehicle(self) -> None:
        """重置指定无人机到起飞点（不影响其他无人机）。"""
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

        pose = airsim.Pose(
            position_val=airsim.Vector3r(
                float(self.start_pos[0]),
                float(self.start_pos[1]),
                float(self.start_pos[2]),
            ),
            orientation_val=airsim.to_quaternion(0.0, 0.0, 0.0),
        )
        self.client.simSetVehiclePose(
            pose, ignore_collision=True, vehicle_name=self.vehicle_name
        )

        time.sleep(0.08)
        self.client.moveByVelocityAsync(
            0.0, 0.0, 0.0, duration=0.1, vehicle_name=self.vehicle_name
        ).join()
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(0.5)

    def _sample_target(self) -> np.ndarray:
        """在前方区域随机采样终点。"""
        return self.np_random.uniform(self.target_min, self.target_max).astype(
            np.float32
        )

    def _get_kinematics(self) -> tuple[np.ndarray, np.ndarray]:
        """读取指定无人机的位置与速度（NED 坐标系）。"""
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        kin = state.kinematics_estimated

        position = np.array(
            [kin.position.x_val, kin.position.y_val, kin.position.z_val],
            dtype=np.float32,
        )
        velocity = np.array(
            [
                kin.linear_velocity.x_val,
                kin.linear_velocity.y_val,
                kin.linear_velocity.z_val,
            ],
            dtype=np.float32,
        )
        return position, velocity

    def _compute_distance(self, pos: np.ndarray, target: np.ndarray) -> float:
        return float(np.linalg.norm(target - pos))

    def _get_depth_feature(self) -> np.ndarray:
        """获取指定无人机的深度图特征。"""
        return get_depth_feature_for_vehicle(
            self.client, self.depth_feature_size, self.max_depth_m, self.vehicle_name
        )

    def _get_obs(self) -> np.ndarray:
        current_pos, current_vel = self._get_kinematics()
        target_delta = (self.target_pos - current_pos).astype(np.float32)
        depth_feat = self._get_depth_feature()
        obs = np.concatenate([depth_feat, current_vel, target_delta], axis=0).astype(
            np.float32
        )
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        del options
        super().reset(seed=seed)

        self._safe_reset_vehicle()
        self.target_pos = self._sample_target()
        current_pos, _ = self._get_kinematics()

        self.prev_distance = self._compute_distance(current_pos, self.target_pos)
        self.current_step = 0
        self.v_cmd_prev = np.zeros(3, dtype=np.float32)
        self.v_cmd_prev_last = np.zeros(3, dtype=np.float32)

        obs = self._get_obs()
        info = {
            "target_pos": self.target_pos.copy(),
            "distance_to_target": self.prev_distance,
        }
        return obs, info

    def _log_event(self, event, reward, distance):
        print(
            f"[{self.vehicle_name}] step={self.current_step} "
            f"event={event} reward={reward:.2f} dist={distance:.2f}"
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步环境交互。

        流程：
        1) 将策略输出动作限制到 [-1, 1]，并映射为速度指令；
        2) 在 AirSim 中下发速度控制并等待一个 step_duration；
        3) 读取碰撞/位置/速度，计算到目标点距离；
        4) 根据碰撞、到达目标、距离变化计算奖励并判定终止；
        5) 构造下一时刻观测与诊断信息并返回。
        """
        # 记录环境步数，用于最大步长截断逻辑。
        self.current_step += 1

        # 将输入动作标准化为期望形状 (3,)，并裁剪到动作空间范围 [-1, 1]。
        # 这样可以容忍策略网络输出轻微越界，避免向模拟器发送异常指令。
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape) # type: ignore
        action = np.clip(action, self.action_space.low, self.action_space.high) # type: ignore

        # 将归一化动作按最大速度线性放缩为 NED 坐标系速度命令。
        # vx, vy, vz 分别对应前后、左右、上下方向速度（单位 m/s）。
        v_target = action * self.max_speed
        v_cmd = self.v_cmd_prev + self.alpha * (v_target - self.v_cmd_prev)
        self.v_cmd_prev_last = self.v_cmd_prev.copy()
        self.v_cmd_prev = v_cmd.copy()
        vx, vy, vz = v_cmd.tolist()

        # 异步下发速度控制。这里给出略大于 step_duration 的控制时长，
        # 目的是让无人机在当前步内持续执行该动作，降低控制空窗影响。
        self.client.moveByVelocityAsync(
            float(vx),
            float(vy),
            float(vz),
            duration=self.step_duration * 1.25,
            vehicle_name=self.vehicle_name,
        )
        # 训练环境采用“离散步推进”，通过 sleep 对齐一个 RL step 的物理时长。
        time.sleep(self.step_duration)

        # 执行动作后立刻读取碰撞状态与运动学信息，作为本步转移结果。
        collision = self.client.simGetCollisionInfo(
            vehicle_name=self.vehicle_name
        ).has_collided
        current_pos, current_vel = self._get_kinematics()
        distance_to_target = self._compute_distance(current_pos, self.target_pos)

        # Gymnasium 语义：
        # terminated 表示任务自然结束（成功/失败）；
        # truncated 表示因外部限制中断（如超步数）。
        terminated = False
        truncated = False

        progress = self.prev_distance - distance_to_target

        event = None
        if collision:
            reward = -100.0
            terminated = True
            event = "collision"
        elif distance_to_target < self.success_threshold:
            reward = 100.0
            terminated = True
            event = "success"
        else:
            # 非对称进度奖励：正向加倍鼓励接近，负向封顶鼓励探索
            if progress >= 0:
                r_progress = progress * 2.0
            else:
                r_progress = max(progress, -0.5)
            reward = float(r_progress + 0.03)

        # 超过最大步数时触发截断（前提是尚未自然终止）。
        if self.current_step >= self.max_episode_steps and not terminated:
            truncated = True

        # 更新“上一时刻距离”，供下一步计算距离改变量。
        self.prev_distance = distance_to_target

        # 组装下一观测。
        obs = self._get_obs()

        # self._log_event(event, reward, distance_to_target)

        # info 用于训练外的调试与可视化分析，不参与策略梯度计算。
        info = {
            "target_pos": self.target_pos.copy(),
            "position": current_pos,
            "velocity": current_vel,
            "distance_to_target": distance_to_target,
            "collision": bool(collision),
            # raw_action 是裁剪后的归一化动作；scaled_velocity_cmd 为实际下发速度。
            "raw_action": action.copy(),
            "scaled_velocity_cmd": np.array([vx, vy, vz], dtype=np.float32),
            "event": event,
            "reward": float(reward),
            "episode_step": self.current_step
        }
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
            self.client.landAsync(vehicle_name=self.vehicle_name).join()
        except Exception:
            pass
        finally:
            try:
                self.client.armDisarm(False, vehicle_name=self.vehicle_name)
                self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
            except Exception:
                pass
