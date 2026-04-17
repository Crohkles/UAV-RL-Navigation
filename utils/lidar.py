import airsim
import numpy as np
import cv2
import time


class HighPrecisionCylindricalProcessor:
    def __init__(self, width=1024, height=256, max_dist=20.0):
        self.width = width
        self.height = height
        self.max_dist = max_dist
        # 建议根据 AirSim settings.json 中的垂直 FOV 修改
        self.v_fov_upper = 10  # 向上 10 度
        self.v_fov_lower = -30  # 向下 30 度 (通常无人机向下看更多)
        self.v_fov_total = self.v_fov_upper - self.v_fov_lower

    def process_lidar(self, lidar_points):
        # 1. 坐标提取 (AirSim: x=前, y=右, z=下)
        # 为了符合直觉，我们将 z 取反，变为“向上为正”
        x = lidar_points[:, 0]
        y = lidar_points[:, 1]
        z = -lidar_points[:, 2]

        # 2. 计算极坐标
        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        # 水平角：arctan2(y, x) 范围 [-pi, pi]
        h_angle = np.arctan2(y, x)
        # 垂直角：arcsin(z / dist)
        v_angle = np.arcsin(z / dist)

        # 3. 映射到图像坐标
        # 水平映射：将 [-pi, pi] 映射到 [0, width-1]
        u_f = (h_angle + np.pi) / (2 * np.pi) * (self.width - 1)

        # 垂直映射：根据 FOV 限制映射到 [0, height-1]
        v_fov_upper_rad = np.deg2rad(self.v_fov_upper)
        v_fov_lower_rad = np.deg2rad(self.v_fov_lower)
        v_fov_total_rad = np.deg2rad(self.v_fov_total)

        # 翻转垂直坐标，使得上方在图像顶部
        v_f = (v_fov_upper_rad - v_angle) / v_fov_total_rad * (self.height - 1)

        # 4. 过滤越界点和过远点
        valid = (u_f >= 0) & (u_f < self.width) & \
                (v_f >= 0) & (v_f < self.height) & \
                (dist < self.max_dist)

        u_v = u_f[valid].astype(int)
        v_v = v_f[valid].astype(int)
        d_v = dist[valid]

        # 5. Z-buffer 填充
        # 初始化为 max_dist（背景色）
        img = np.full((self.height, self.width), self.max_dist, dtype=np.float32)

        # 排序：先画远的，后画近的，确保近处遮挡远处
        idx = np.argsort(-d_v)
        img[v_v[idx], u_v[idx]] = d_v[idx]

        # 6. 后处理增强
        # 由于 Lidar 是线性的，点非常稀疏，必须通过形态学操作“加粗”点云
        # 否则 cv2.imshow 几乎看不见内容
        kernel = np.ones((3, 3), np.uint8)
        img_dilated = cv2.dilate(img, kernel, iterations=1)

        # 仅对有值的区域进行平滑
        mask = (img < self.max_dist).astype(np.uint8)
        img_final = cv2.bilateralFilter(img, d=5, sigmaColor=0.5, sigmaSpace=5)

        return img, mask


# --- 主程序 ---
# 建议宽度远大于高度，因为是 360 度环视
processor = HighPrecisionCylindricalProcessor(width=1024, height=256, max_dist=40.0)
client = airsim.MultirotorClient()
client.confirmConnection()

print("按下 'q' 退出...")
last_time = time.time()

while True:
    # 频率控制：1秒刷新一次
    current_time = time.time()
    if current_time - last_time < 1.0:
        time.sleep(0.01)
        continue
    last_time = current_time

    data = client.getLidarData(lidar_name="LidarSensor1")

    if len(data.point_cloud) >= 3:
        points = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)

        depth_map, mask = processor.process_lidar(points)

        # --- 可视化优化 ---
        # 1. 归一化深度值 (0 为近, 1 为远)
        vis_img = np.clip(depth_map / processor.max_dist, 0, 1)
        # 2. 反转：让近处更亮，远处更暗（符合人类视觉）
        vis_img = 1.0 - vis_img
        # 3. 增强对比度：只在有数据的地方显示
        vis_img = (vis_img * 255).astype(np.uint8)

        # 4. 使用 COLORMAP_MAGMA 或 JET 增加深度感
        vis_color = cv2.applyColorMap(vis_img, cv2.COLORMAP_MAGMA)
        # 将没有点云的地方设为黑色
        vis_color[mask == 0] = 0

        cv2.imshow("360 Degree Cylindrical Projection", vis_color)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()