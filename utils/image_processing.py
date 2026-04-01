import numpy as np
import airsim

def get_depth_feature(
    client: airsim.MultirotorClient, 
    depth_feature_size: tuple[int, int], 
    max_depth_m: float
) -> np.ndarray:
    """获取深度图并降采样为一维特征向量。"""
    depth_dim = depth_feature_size[0] * depth_feature_size[1]
    try:
        responses = client.simGetImages(
            [
                airsim.ImageRequest(
                    "FrontCamera", airsim.ImageType.DepthPlanar, True, False
                )
            ]
        )
        if not responses:
            return np.zeros(depth_dim, dtype=np.float32)

        response = responses[0]
        if response.height <= 0 or response.width <= 0:
            return np.zeros(depth_dim, dtype=np.float32)

        depth_flat = np.asarray(response.image_data_float, dtype=np.float32)
        if depth_flat.size != response.height * response.width:
            return np.zeros(depth_dim, dtype=np.float32)

        depth_img = depth_flat.reshape(response.height, response.width)
        depth_img = np.nan_to_num(
            depth_img,
            nan=max_depth_m,
            posinf=max_depth_m,
            neginf=0.0,
        )
        depth_img = np.clip(depth_img, 0.0, max_depth_m)
        depth_img = depth_img / max_depth_m

        # 用等距采样把任意分辨率压到固定，便于和速度/目标向量拼接。
        h_idx = np.linspace(
            0, depth_img.shape[0] - 1, depth_feature_size[0]
        ).astype(np.int32)
        w_idx = np.linspace(
            0, depth_img.shape[1] - 1, depth_feature_size[1]
        ).astype(np.int32)
        depth_small = depth_img[np.ix_(h_idx, w_idx)]
        return depth_small.reshape(-1).astype(np.float32)
    except Exception:
        return np.zeros(depth_dim, dtype=np.float32)