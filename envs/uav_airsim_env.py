import time
from typing import Optional

import airsim
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils.image_processing import get_depth_feature

class UAVSimpleTrainEnv(gym.Env):
	"""用于UAV避障与导航初训的简化Gymnasium环境。"""

	metadata = {"render_modes": []}

	def __init__(
		self,
		max_speed: float = 5.0,
		step_duration: float = 0.5,
		max_episode_steps: int = 200,
		success_threshold: float = 2.0,
		depth_feature_size: tuple[int, int] = (12, 12),
		max_depth_m: float = 50.0,
	) -> None:
		super().__init__()

		self.max_speed = float(max_speed)
		self.step_duration = float(step_duration)
		self.max_episode_steps = int(max_episode_steps)
		self.success_threshold = float(success_threshold)
		self.depth_feature_size = depth_feature_size
		self.max_depth_m = float(max_depth_m)

		# AirSim 默认使用 NED 坐标系：x北向、y东向、z向下为正。
		self.start_pos = np.array([0.0, 0.0, -30.0], dtype=np.float32)
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

	def _connect_client(self) -> None:
		"""建立连接并获取控制权。"""
		self.client.confirmConnection()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)

	def _safe_reset_vehicle(self) -> None:
		"""按回合重置无人机，并强制传送到固定起飞点。"""
		self.client.reset()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)

		pose = airsim.Pose(
			position_val=airsim.Vector3r(
				float(self.start_pos[0]),
				float(self.start_pos[1]),
				float(self.start_pos[2]),
			),
			orientation_val=airsim.to_quaternion(0.0, 0.0, 0.0),
		)
		self.client.simSetVehiclePose(pose, ignore_collision=True)

		# 给予仿真器少量时间，避免瞬时状态读数抖动。
		time.sleep(0.08)
		# 极短时间内给一个速度为0的指令，消除重置前残留的惯性并避免下坠。
		self.client.moveByVelocityAsync(0.0, 0.0, 0.0, duration=0.1).join()
		self.client.hoverAsync().join()
		time.sleep(0.5)

	def _sample_target(self) -> np.ndarray:
		"""在前方区域随机采样终点。"""
		return self.np_random.uniform(self.target_min, self.target_max).astype(np.float32)

	def _get_kinematics(self) -> tuple[np.ndarray, np.ndarray]:
		"""读取位置与速度（均在NED坐标系）。"""
		state = self.client.getMultirotorState()
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
		"""获取深度图并降采样为一维特征向量。"""
		return get_depth_feature(self.client, self.depth_feature_size, self.max_depth_m)

	def _get_obs(self) -> np.ndarray:
		"""
		状态拼接顺序：
		1) 深度特征 144维
		2) 当前速度 3维
		3) 目标相对位移向量 3维
		"""
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

		obs = self._get_obs()
		info = {
			"target_pos": self.target_pos.copy(),
			"distance_to_target": self.prev_distance,
		}
		return obs, info

	def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
		self.current_step += 1

		action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
		action = np.clip(action, self.action_space.low, self.action_space.high)

		# TD3 输出为 [-1,1]，缩放后映射到最大速度指令。
		vx, vy, vz = (action * self.max_speed).tolist()
		self.client.moveByVelocityAsync(
			float(vx), float(vy), float(vz), duration=self.step_duration*1.25
		)
		time.sleep(self.step_duration)

		collision = self.client.simGetCollisionInfo().has_collided
		current_pos, current_vel = self._get_kinematics()
		distance_to_target = self._compute_distance(current_pos, self.target_pos)

		terminated = False
		truncated = False

		if collision:
			reward = -100.0
			terminated = True
		elif distance_to_target < self.success_threshold:
			reward = 100.0
			terminated = True
		else:
			# Dense Reward: 靠近目标的距离奖励 + 每步微小生存奖励。
			progress_reward = self.prev_distance - distance_to_target
			if progress_reward >= 0:
				progress_reward *= 2.0
			else:
				# 减小远离惩罚
				progress_reward = max(progress_reward, -0.5)
			
			reward = float(progress_reward + 0.03)

		if self.current_step >= self.max_episode_steps and not terminated:
			truncated = True

		self.prev_distance = distance_to_target
		obs = self._get_obs()
		info = {
			"target_pos": self.target_pos.copy(),
			"position": current_pos,
			"velocity": current_vel,
			"distance_to_target": distance_to_target,
			"collision": bool(collision),
			"raw_action": action.copy(),
			"scaled_velocity_cmd": np.array([vx, vy, vz], dtype=np.float32),
		}
		return obs, float(reward), terminated, truncated, info

	def close(self) -> None:
		try:
			self.client.hoverAsync().join()
			self.client.landAsync().join()
		except Exception:
			pass
		finally:
			try:
				self.client.armDisarm(False)
				self.client.enableApiControl(False)
			except Exception:
				pass

