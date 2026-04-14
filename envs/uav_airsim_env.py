import time
import math
from typing import Optional

import airsim
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2

from utils.image_processing import get_depth_feature

class UAVSimpleTrainEnv(gym.Env):
	"""用于UAV避障与导航初训的简化Gymnasium环境。"""

	metadata = {"render_modes": []}

	def __init__(
		self,
		vehicle_name = 'SimpleFlight',
		max_speed: float = 5.0,
		step_duration: float = 0.5,
		max_episode_steps: int = 200,
		success_threshold: float = 2.0,
		depth_feature_size: tuple[int, int] = (12, 12),
		max_depth_m: float = 50.0,
	) -> None:
		super().__init__()

		self.vehicle_name = vehicle_name

		self.max_speed = float(max_speed)
		self.step_duration = float(step_duration)
		self.max_episode_steps = int(max_episode_steps)
		self.success_threshold = float(success_threshold)
		self.depth_feature_size = depth_feature_size
		self.max_depth_m = float(max_depth_m)

		self.diff_x_max = 65
		self.diff_y_max = 65
		self.diff_z_max = 5

		self.v_x_max = 7.0
		self.v_y_max = 7.0
		self.v_z_max = 2.0

		self.yaw_rate_max_rad = 0.2

		self.min_distance_to_obstacles = 5
		self.previous_distance_from_des_point = 0

		# AirSim 默认使用 NED 坐标系：x北向、y东向、z向下为正。
		self.start_pos = np.array([0.0, 0.0, -30.0], dtype=np.float32)
		self.target_min = np.array([10.0, -15.0, -40.0], dtype=np.float32)
		self.target_max = np.array([30.0, 15.0, -20.0], dtype=np.float32)

		self.action_space = spaces.Box(
            low=np.array([
                -self.v_x_max, -self.v_y_max, -self.v_z_max, -self.yaw_rate_max_rad]),
            high=np.array([
                self.v_x_max, self.v_y_max, self.v_z_max, self.yaw_rate_max_rad]),
            dtype=np.float32
        )


		self.screen_width = 80
		self.screen_height = 100
		self.max_depth_meters = 50

		self.observation_space = spaces.Box(
			low=0.0,
			high=255.0,
			shape=(self.screen_height, self.screen_width, 2),
			dtype=np.uint8,
		)

		self.client = airsim.MultirotorClient(port=41454)
		self._connect_client()

		self.target_pos = np.zeros(3, dtype=np.float32)
		self.prev_distance = 0.0
		self.current_step = 0

		self.previous_action = np.zeros(4, dtype=np.float32)
		self.current_action = np.zeros(4, dtype=np.float32)

	def _connect_client(self) -> None:
		"""建立连接并获取控制权。"""
		self.client.confirmConnection()
		self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
		self.client.armDisarm(True, vehicle_name=self.vehicle_name)

	def _safe_reset_vehicle(self) -> None:
		"""按回合重置无人机，并强制传送到固定起飞点。"""
		self.client.reset()
		self.client.simPause(False)
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
		state = self.client.getMultirotorState(self.vehicle_name)
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
				kin.angular_velocity.z_val,
			],
			dtype=np.float32,
		)
		return position, velocity

	def _compute_distance(self, pos: np.ndarray, target: np.ndarray) -> float:
		return float(np.linalg.norm(target - pos))

	def _get_depth_feature(self) -> np.ndarray:
		"""获取深度图并降采样为一维特征向量。"""
		return get_depth_feature(self.client, self.depth_feature_size, self.max_depth_m)

	def _get_depth_image(self, camera_name):
		"""
		Continuously request a depth image from the specified camera until a valid image is received.
		Returns:
			depth_meter (ndarray): Depth image in meters.
		"""
		while True:
			try:
				responses = self.client.simGetImages([
					airsim.ImageRequest(camera_name, airsim.ImageType.DepthVis, True)
				])

				if responses[0].width != 0:
					depth_img = airsim.list_to_2d_float_array(
						responses[0].image_data_float, responses[0].width, responses[0].height)
					depth_meter = depth_img * 100
					return depth_meter
				else:
					print("get_image_fail...")

			except Exception as e:
				print(f"Unexpected error: {e}. Retrying...")

			time.sleep(0.1)

	def _get_attitude(self):
		orientation = self.client.simGetVehiclePose(self.vehicle_name).orientation
		return airsim.to_eularian_angles(orientation)

	def _get_relative_yaw(self):
		"""
		Compute the relative yaw between the current position and the goal.

		Returns:
			yaw_error (float): Angular difference from UAV heading to goal in radians within [-pi, pi].
		"""
		current_position, _ = self._get_kinematics()
		dx = self.target_pos[0] - current_position[0]
		dy = self.target_pos[1] - current_position[1]
		angle_to_goal = math.atan2(dy, dx)

		yaw_current = self._get_attitude()[2]
		yaw_error = angle_to_goal - yaw_current

		if yaw_error > math.pi:
			yaw_error -= 2 * math.pi
		elif yaw_error < -math.pi:
			yaw_error += 2 * math.pi

		return yaw_error

	def _get_state_feature(self):
		"""
		Update and retrieve the current UAV state in normalized form.

		Returns:
			state_norm (np.ndarray): Normalized state vector in the range [0, 255].
		"""
		# Get current position
		current_position, velocity = self._get_kinematics()
		x, y, z = current_position

		# Compute distance difference to goal
		dx = self.target_pos[0] - x
		dy = self.target_pos[1] - y
		dz = self.target_pos[2] - z

		# Get velocity and angular velocity (yaw rate)
		velocity_x, velocity_y, velocity_z, yaw_rate = velocity

		# Get current pitch, roll, yaw
		pitch, roll, yaw = self._get_attitude()

		# Compute relative yaw between current heading and goal
		relative_yaw = self._get_relative_yaw()

		# Normalize each part of the state vector (scaled to [0, 255])
		distance_diff_norm = np.array([dx / self.diff_x_max, dy / self.diff_y_max, dz / self.diff_z_max])
		distance_diff_norm = (distance_diff_norm + 1) * 255 / 2

		velocity_norm = np.array([velocity_x / self.v_x_max, velocity_y / self.v_y_max, velocity_z / self.v_z_max])
		velocity_norm = (velocity_norm + 1) * 255 / 2

		relative_yaw_norm = (relative_yaw / (math.pi / 2) + 1) * 255 / 2
		yaw_rate_norm = (yaw_rate / self.yaw_rate_max_rad + 1) * 255 / 2

		# Store raw state (mostly for logging/debugging)
		self.state_raw = np.array([
			x, y, z, dx, dy, dz,
			velocity_x, velocity_y, velocity_z,
			math.degrees(pitch), math.degrees(roll), math.degrees(yaw),
			math.degrees(relative_yaw), math.degrees(yaw_rate)
		])

		# Construct normalized state vector based on environment and privilege level
		state_norm = np.concatenate((distance_diff_norm, velocity_norm, [relative_yaw_norm, yaw_rate_norm]))

		state_norm = np.clip(state_norm, 0, 255)  # Ensure range is within [0, 255]
		self.state_norm = state_norm

		return state_norm

	def _get_obs(self) -> np.ndarray:
		"""
		Get depth image and embed state information as a 2-channel observation.
		Channel 0: Processed depth image
		Channel 1: State feature array embedded at top-left corner
		"""
		image = self._get_depth_image("FrontDepthCamera")
		image_resize = cv2.resize(image, (self.screen_width, self.screen_height))
		self.min_distance_to_obstacles = image.min()
		image_scaled = np.clip(image_resize, 0, self.max_depth_meters) / self.max_depth_meters * 255
		image_scaled = 255 - image_scaled
		image_uint8 = image_scaled.astype(np.uint8)

		state_feature_array = np.zeros((self.screen_height, self.screen_width))
		state_feature = self._get_state_feature()
		state_feature_array[0, 0:8] = state_feature

		image_with_state = np.array([image_uint8, state_feature_array])
		image_with_state = image_with_state.swapaxes(0, 2).swapaxes(0, 1)

		self.feature_all = image_with_state
		return image_with_state

	def _set_action(self, action):
		"""
		Apply a velocity and yaw rate action to the drone in AirSim.

		Args:
			action (array-like): [v_x, v_y, v_z, yaw_rate]
		"""
		self.previous_action = self.current_action

		# Parse and apply new action
		v_x, v_y, v_z, yaw_rate = map(float, action)
		self.current_action = np.array([v_x, v_y, v_z, yaw_rate])

		# Issue movement command to simulator
		self.client.simPause(False)
		self.client.moveByVelocityAsync(
			v_x, v_y, v_z, duration=0.1,
			drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
			yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate)),
			vehicle_name=self.vehicle_name
		).join()
		self.client.simPause(True)

	def _get_distance_to_goal_3d(self):
		current_pos, _ = self._get_kinematics()
		goal_pos = self.target_pos
		dx = current_pos[0] - goal_pos[0]
		dy = current_pos[1] - goal_pos[1]
		dz = current_pos[2] - goal_pos[2]

		return math.sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2))

	def _get_vector_angle(self):
		velocity, _ = self._get_kinematics()
		angle = math.atan2(velocity[1], velocity[0])
		yaw_current = self._get_attitude()[2]
		yaw_error = angle - yaw_current

		if yaw_error > math.pi:
			yaw_error -= 2 * math.pi
		elif yaw_error < -math.pi:
			yaw_error += 2 * math.pi
		
		return yaw_error

	def _compute_reward(self, done, is_success):
		"""
		Lightweight reward function considering only distance to goal, orientation error, and obstacle proximity.
		"""
		reward = 0
		reward_reach = 10
		reward_crash = -5
		# reward_outside = -5
		if not done:
			# 1. Reward for reducing distance
			distance_now = self._get_distance_to_goal_3d()
			reward_distance = self.previous_distance_from_des_point - distance_now
			reward_distance = np.clip(reward_distance, -1, 1)
			self.previous_distance_from_des_point = distance_now

			# 2. Obstacle distance penalty
			if self.min_distance_to_obstacles < 4:
				punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - 1) / 3, 0, 1)
			else:
				punishment_obs = 0

			# 3. Orientation error penalty
			error_now = self._get_vector_angle()
			punishment_angle = abs(np.clip(error_now / (2 * math.pi), -1, 1))

			# Final reward combination
			reward = 5.0 * reward_distance - punishment_obs - punishment_angle
			reward = np.clip(reward / 3, -1, 1)
			reward_list = [5.0 * reward_distance, - punishment_obs, - punishment_angle]
		else:
			reward_list = [0, 0, 0]
			if self.current_step < 50:
				reward = -10
			elif is_success:
				reward = reward_reach
			# elif self.is_not_inside_fly_range():
			# 	reward = reward_outside
			else:
				reward = reward_crash

		return float(reward), reward_list

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

	# def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
	# 	self.current_step += 1

	# 	action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
	# 	action = np.clip(action, self.action_space.low, self.action_space.high)

	# 	# TD3 输出为 [-1,1]，缩放后映射到最大速度指令。
	# 	vx, vy, vz = (action * self.max_speed).tolist()
	# 	self.client.moveByVelocityAsync(
	# 		float(vx), float(vy), float(vz), duration=self.step_duration*1.25
	# 	)
	# 	time.sleep(self.step_duration)

	# 	collision = self.client.simGetCollisionInfo().has_collided
	# 	current_pos, current_vel = self._get_kinematics()
	# 	distance_to_target = self._compute_distance(current_pos, self.target_pos)

	# 	terminated = False
	# 	truncated = False

	# 	if collision:
	# 		reward = -100.0
	# 		terminated = True
	# 	elif distance_to_target < self.success_threshold:
	# 		reward = 100.0
	# 		terminated = True
	# 	else:
	# 		# Dense Reward: 靠近目标的距离奖励 + 每步微小生存奖励。
	# 		progress_reward = self.prev_distance - distance_to_target
	# 		if progress_reward >= 0:
	# 			progress_reward *= 2.0
	# 		else:
	# 			# 减小远离惩罚
	# 			progress_reward = max(progress_reward, -0.5)
			
	# 		reward = float(progress_reward + 0.03)

	# 	if self.current_step >= self.max_episode_steps and not terminated:
	# 		truncated = True

	# 	self.prev_distance = distance_to_target
	# 	obs = self._get_obs()
	# 	info = {
	# 		"target_pos": self.target_pos.copy(),
	# 		"position": current_pos,
	# 		"velocity": current_vel,
	# 		"distance_to_target": distance_to_target,
	# 		"collision": bool(collision),
	# 		"raw_action": action.copy(),
	# 		"scaled_velocity_cmd": np.array([vx, vy, vz], dtype=np.float32),
	# 	}
	# 	return obs, float(reward), terminated, truncated, info

	def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
		"""
		Execute one environment step using the given action.
		"""
		# Apply action to the UAV
		self._set_action(action)

		# Get the updated observation
		obs = self._get_obs()

		current_pos, current_vel = self._get_kinematics()
		distance_to_target = self._compute_distance(current_pos, self.target_pos)

		# Determine task status depending on environment name
		is_crashed = self.client.simGetCollisionInfo().has_collided
		is_success = distance_to_target < self.success_threshold

		info = {
			'is_success': is_success,
			'is_crash': is_crashed,
			# 'is_not_in_workspace': self.is_not_inside_fly_range(),
	 		"target_pos": self.target_pos.copy(),
			'step_num': self.current_step
		}

		terminated = is_crashed or is_success
		truncated = self.current_step >= self.max_episode_steps \
			and not terminated
		
		# Handle success/failure statistics and curriculum adjustment
		if terminated or truncated:
			print(info)

		# Compute reward using selected reward function
		reward, _= self._compute_reward(terminated or truncated,\
											  is_success)

		# Update counters
		print(
			f"\rStep: {self.current_step:>4d}/{self.max_episode_steps}  "
			f"Dist: {distance_to_target:>7.2f}m  "
			f"Reward: {reward:>+7.3f}",
			end="", flush=True,
		)
		if terminated or truncated:
			print()  # episode end: newline so next episode starts fresh
		self.current_step += 1

		return obs, reward, terminated, truncated, info

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

