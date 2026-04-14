import airsim
import numpy as np
import cv2


class LidarToCylindricalProcessor:
    def __init__(self, width, height, max_dist):
        self.width = width  # 360度全景图的宽度
        self.height = height  # 垂直分辨率
        self.max_dist = max_dist
        # 这里的垂直FOV需要根据AirSim settings.json中的Lidar配置来修改
        self.v_fov_upper = 20  # 垂直上视角 (度)
        self.v_fov_lower = -20  # 垂直下视角 (度)
        self.v_fov_total = self.v_fov_upper - self.v_fov_lower

    def process_lidar(self, lidar_points):
        # 1. 计算极坐标
        x = lidar_points[:, 0]
        y = lidar_points[:, 1]
        z = lidar_points[:, 2]

        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        # 水平角 h_angle 范围: [-pi, pi]
        h_angle = np.arctan2(y, x)
        # 垂直角 v_angle 范围: [v_fov_lower, v_fov_upper] (弧度)
        v_angle = np.arcsin(z / dist)

        # 2. 映射到像素坐标
        # 水平映射：[-pi, pi] -> [0, width-1]
        u = (0.5 * (1 - h_angle / np.pi) * (self.width - 1)).astype(int)

        # 垂直映射：[v_fov_lower, v_fov_upper] -> [0, height-1]
        v_fov_lower_rad = np.deg2rad(self.v_fov_lower)
        v_fov_total_rad = np.deg2rad(self.v_fov_total)
        v = ((1 - (v_angle - v_fov_lower_rad) / v_fov_total_rad) * (self.height - 1)).astype(int)

        # 3. 过滤无效点
        valid_mask = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        dist_valid = dist[valid_mask]

        # 4. 生成深度图 (初始化为最大距离)
        img = np.full((self.height, self.width), self.max_dist, dtype=np.float32)

        # 填充像素（如果有多个点落在同一个像素，保留最近的点）
        # 使用 argsort 确保远处的点先画，近处的点后画并覆盖
        indices = np.argsort(-dist_valid)
        img[v_valid[indices], u_valid[indices]] = dist_valid[indices]

        # 5. 后处理：简单的形态学膨胀以填补激光线束间的空隙
        # 使用最小值池化性质的腐蚀操作，因为障碍物是“小值”
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel)

        return img


# --- 使用示例 ---
processor = LidarToCylindricalProcessor(width=1024, height=256, max_dist=15.0)
client = airsim.MultirotorClient()

data = client.getLidarData(lidar_name="LidarSensor1")

if len(data.point_cloud) >= 3:
    points = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)

    cylindrical_img = processor.process_lidar(points)

    # 可视化
    # 归一化到 0-1 方便显示，并使用伪彩色增强对比度
    vis_img = np.clip(cylindrical_img / 15.0, 0, 1)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    cv2.imshow("Cylindrical Lidar Projection", vis_color)


cv2.destroyAllWindows()