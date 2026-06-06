import heapq
import json
import math
import os
from typing import Iterable, List, Sequence, Tuple, TypeVar

import numpy as np


Pixel = Tuple[int, int]
PixelWaypoint3D = Tuple[int, int, float]
Waypoint = Tuple[float, float, float]
PathPoint = TypeVar("PathPoint")


class AStarPlanner:
    """A* planner over a height-map occupancy grid.

    self.grid is True for blocked cells and False for free cells. Pixel
    coordinates are always (u, v), where u is the image column and v is the row.
    """

    def __init__(
        self,
        map_path: str,
        obstacle_height_threshold: float = 25.0,
        height_channel: str = "r",
        clearance_cost_weight: float = 0.0,
        clearance_cost_radius: float = 0.0,
    ):
        self.map_path = map_path
        self.obstacle_height_threshold = float(obstacle_height_threshold)
        self.height_channel = height_channel
        self.clearance_cost_weight = max(0.0, float(clearance_cost_weight))
        self.clearance_cost_radius = max(0.0, float(clearance_cost_radius))

        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "AStarPlanner requires OpenCV. Install opencv-python in the "
                "environment used to run map planning."
            ) from exc

        self._cv2 = cv2
        raw_img = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            raise ValueError(
                "Cannot read map file: {}. Check the path and image format.".format(
                    map_path
                )
            )

        self.raw_img = raw_img
        self.height_map = self._extract_height_map(raw_img)
        self.grid = self.height_map >= self.obstacle_height_threshold
        self.clearance_map = self._compute_clearance_map()

    def _compute_clearance_map(self):
        free_mask = (~self.grid).astype(np.uint8)
        return self._cv2.distanceTransform(free_mask, self._cv2.DIST_L2, 5)

    def _extract_height_map(self, image):
        if image.ndim == 2:
            return image.astype(np.float32)

        channel = str(self.height_channel).lower()
        channel_index = {
            "b": 0,
            "blue": 0,
            "g": 1,
            "green": 1,
            "r": 2,
            "red": 2,
            "a": 3,
            "alpha": 3,
        }.get(channel)

        if channel_index is None:
            raise ValueError("unsupported height channel: {}".format(self.height_channel))
        if channel_index >= image.shape[2]:
            raise ValueError(
                "height channel {} is unavailable for image shape {}".format(
                    self.height_channel, image.shape
                )
            )

        return image[:, :, channel_index].astype(np.float32)

    @property
    def width(self):
        return int(self.grid.shape[1])

    @property
    def height(self):
        return int(self.grid.shape[0])

    def in_bounds(self, pixel: Pixel):
        u, v = pixel
        return 0 <= u < self.width and 0 <= v < self.height

    def is_obstacle(self, pixel: Pixel):
        if not self.in_bounds(pixel):
            return True
        u, v = pixel
        return bool(self.grid[v, u])

    def validate_free_pixel(self, pixel: Pixel, label: str):
        if not self.in_bounds(pixel):
            raise ValueError(
                "{} pixel {} is outside map bounds {}x{}".format(
                    label, pixel, self.width, self.height
                )
            )
        if self.is_obstacle(pixel):
            raise ValueError("{} pixel {} is inside an obstacle".format(label, pixel))

    def clearance_at(self, pixel: Pixel) -> float:
        if not self.in_bounds(pixel):
            return 0.0
        u, v = pixel
        return float(self.clearance_map[v, u])

    def clearance_cost(self, pixel: Pixel) -> float:
        if self.clearance_cost_weight <= 0.0 or self.clearance_cost_radius <= 0.0:
            return 0.0

        clearance = self.clearance_at(pixel)
        if clearance >= self.clearance_cost_radius:
            return 0.0

        deficit = (self.clearance_cost_radius - clearance) / self.clearance_cost_radius
        return self.clearance_cost_weight * deficit * deficit

    @staticmethod
    def heuristic(a: Pixel, b: Pixel):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def astar(
        self,
        start: Pixel,
        goal: Pixel,
        allow_diagonal: bool = True,
        prevent_corner_cutting: bool = True,
        verbose: bool = False,
    ) -> List[Pixel]:
        self.validate_free_pixel(start, "start")
        self.validate_free_pixel(goal, "goal")

        if allow_diagonal:
            neighbors = [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
        else:
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        closed = set()
        came_from = {}
        gscore = {start: 0.0}
        fscore = {start: self.heuristic(start, goal)}
        open_heap = [(fscore[start], start)]

        while open_heap:
            current = heapq.heappop(open_heap)[1]
            if current in closed:
                continue

            if verbose:
                print("scan:", current)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            closed.add(current)
            for du, dv in neighbors:
                neighbor = (current[0] + du, current[1] + dv)
                if self.is_obstacle(neighbor):
                    continue

                if (
                    prevent_corner_cutting
                    and du != 0
                    and dv != 0
                    and (
                        self.is_obstacle((current[0] + du, current[1]))
                        or self.is_obstacle((current[0], current[1] + dv))
                    )
                ):
                    continue

                step_cost = self.heuristic(current, neighbor) + self.clearance_cost(
                    neighbor
                )
                tentative_g = gscore[current] + step_cost
                if neighbor in closed and tentative_g >= gscore.get(neighbor, math.inf):
                    continue

                if tentative_g < gscore.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (fscore[neighbor], neighbor))

        return []

    @staticmethod
    def _reconstruct_path(came_from, current: Pixel):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


class AStarPlanner3D(AStarPlanner):
    """Height-aware A* returning map pixels with dynamic NED flight heights."""

    def __init__(
        self,
        map_path: str,
        obstacle_height_threshold: float = 25.0,
        height_channel: str = "r",
        height_weight: float = 8.0,
        clearance_cost_weight: float = 0.0,
        clearance_cost_radius: float = 0.0,
    ):
        super().__init__(
            map_path=map_path,
            obstacle_height_threshold=obstacle_height_threshold,
            height_channel=height_channel,
            clearance_cost_weight=clearance_cost_weight,
            clearance_cost_radius=clearance_cost_radius,
        )
        self.height_weight = max(0.0, float(height_weight))

    def get_height(self, pixel: Pixel) -> float:
        if not self.in_bounds(pixel):
            return 0.0
        u, v = pixel
        return float(self.height_map[v, u])

    def _transition_cost(self, current: Pixel, neighbor: Pixel) -> float:
        planar_cost = self.heuristic(current, neighbor)
        height_delta = abs(self.get_height(neighbor) - self.get_height(current))
        return (
            planar_cost
            + self.clearance_cost(neighbor)
            + self.height_weight * height_delta
        )

    def astar_3d(
        self,
        start: Pixel,
        goal: Pixel,
        cruise_alt: float,
        height_lift_coef: float = 0.3,
        allow_diagonal: bool = True,
        prevent_corner_cutting: bool = True,
        verbose: bool = False,
    ) -> List[PixelWaypoint3D]:
        self.validate_free_pixel(start, "start")
        self.validate_free_pixel(goal, "goal")

        if allow_diagonal:
            neighbors = [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
        else:
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        closed = set()
        came_from = {}
        gscore = {start: 0.0}
        open_heap = [(self.heuristic(start, goal), start)]

        while open_heap:
            current = heapq.heappop(open_heap)[1]
            if current in closed:
                continue

            if verbose:
                print("scan:", current)

            if current == goal:
                pixel_path = self._reconstruct_path(came_from, current)
                lift = float(height_lift_coef)
                return [
                    (u, v, float(cruise_alt) - lift * self.get_height((u, v)))
                    for u, v in pixel_path
                ]

            closed.add(current)
            for du, dv in neighbors:
                neighbor = (current[0] + du, current[1] + dv)
                if self.is_obstacle(neighbor):
                    continue

                if (
                    prevent_corner_cutting
                    and du != 0
                    and dv != 0
                    and (
                        self.is_obstacle((current[0] + du, current[1]))
                        or self.is_obstacle((current[0], current[1] + dv))
                    )
                ):
                    continue

                tentative_g = gscore[current] + self._transition_cost(current, neighbor)
                if neighbor in closed and tentative_g >= gscore.get(neighbor, math.inf):
                    continue

                if tentative_g < gscore.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (fscore, neighbor))

        return []


def sparsify_path(path: Sequence[PathPoint], stride: int) -> List[PathPoint]:
    if not path:
        return []

    stride = max(1, int(stride))
    sparse = list(path[::stride])
    if sparse[-1] != path[-1]:
        sparse.append(path[-1])
    return sparse


def load_pixel_to_airsim_matrix(matrix_path: str) -> np.ndarray:
    """Load a 3x3 homogeneous pixel-to-AirSim matrix.

    Supported formats:
      - .npy/.npz: numpy arrays
      - .json: either a raw 3x3 list or a dict containing matrix/H/homography
      - .csv/.txt: comma or whitespace separated numeric matrix
    """
    ext = os.path.splitext(matrix_path)[1].lower()

    if ext == ".npy":
        matrix = np.load(matrix_path)
    elif ext == ".npz":
        data = np.load(matrix_path)
        key = "matrix" if "matrix" in data.files else data.files[0]
        matrix = data[key]
    elif ext == ".json":
        with open(matrix_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        matrix = _matrix_from_json_data(data)
    else:
        try:
            matrix = np.loadtxt(matrix_path, delimiter=",")
        except ValueError:
            matrix = np.loadtxt(matrix_path)

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (9,):
        matrix = matrix.reshape(3, 3)
    elif matrix.shape == (2, 3):
        matrix = np.vstack([matrix, np.array([0.0, 0.0, 1.0])])

    if matrix.shape != (3, 3):
        raise ValueError(
            "calibration matrix must be 3x3, got shape {} from {}".format(
                matrix.shape, matrix_path
            )
        )
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("calibration matrix is singular: {}".format(matrix_path))

    return matrix


def _matrix_from_json_data(data):
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise ValueError("JSON matrix file must contain a list or object")

    for key in (
        "pixel_to_airsim_matrix",
        "pixel_to_airsim_homography",
        "homography",
        "matrix",
        "H",
    ):
        if key in data:
            value = data[key]
            if isinstance(value, dict) and "matrix" in value:
                return value["matrix"]
            return value

    # Backward-compatible with the previous calibration helper output.
    affine = data.get("pixel_to_airsim_affine")
    if isinstance(affine, dict) and "matrix" in affine:
        return affine["matrix"]

    raise ValueError(
        "JSON matrix file must contain one of: pixel_to_airsim_matrix, "
        "pixel_to_airsim_homography, homography, matrix, H"
    )


class PixelToAirSimMatrixMapper:
    """Convert map pixels to AirSim x/y with a homogeneous 3x3 matrix."""

    def __init__(self, matrix: Sequence[Sequence[float]]):
        self.matrix = np.asarray(matrix, dtype=np.float64)
        if self.matrix.shape != (3, 3):
            raise ValueError("matrix must be 3x3")
        if abs(np.linalg.det(self.matrix)) < 1e-12:
            raise ValueError("matrix is singular")
        self.inverse_matrix = np.linalg.inv(self.matrix)

    @classmethod
    def from_file(cls, matrix_path: str):
        return cls(load_pixel_to_airsim_matrix(matrix_path))

    @staticmethod
    def _normalize_homogeneous(vector):
        w = float(vector[2])
        if abs(w) < 1e-12:
            raise ValueError("homogeneous coordinate has near-zero scale")
        return float(vector[0] / w), float(vector[1] / w)

    def pixel_to_airsim_xy(self, pixel: Sequence[float]):
        u, v = float(pixel[0]), float(pixel[1])
        airsim_h = self.matrix @ np.array([u, v, 1.0], dtype=np.float64)
        return self._normalize_homogeneous(airsim_h)

    def airsim_xy_to_pixel(self, airsim_xy: Sequence[float]):
        x, y = float(airsim_xy[0]), float(airsim_xy[1])
        pixel_h = self.inverse_matrix @ np.array([x, y, 1.0], dtype=np.float64)
        return self._normalize_homogeneous(pixel_h)

    def pixel_to_waypoint(self, pixel: Sequence[float], z_up: float) -> Waypoint:
        x_air, y_air = self.pixel_to_airsim_xy(pixel)
        return float(x_air), float(y_air), float(z_up)


def pixels_to_waypoints(
    pixels: Iterable[Pixel],
    mapper: PixelToAirSimMatrixMapper,
    z_up: float,
) -> List[Waypoint]:
    return [mapper.pixel_to_waypoint(pixel, z_up) for pixel in pixels]
