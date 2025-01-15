import math

Serial = {
    "port": "COM5",
    "baudrate": 460800,
}
Sensor = {
    "resolution": 8,
    "max_depth": 2000,  ## 10 meters
    "min_depth": 1e-3,
    "output_shape": [256, 256],
    "alpha_edge": math.cos(math.atan(3 / 4 * math.tan(22.5 / 180 * math.pi))),
    "alpha_corner": math.cos(
        math.atan(3 * math.sqrt(2) / 4 * math.tan(22.5 / 180 * math.pi))
    ),
}
h5_cfg = {
    "distance": True,
    "sigma": True,
    "rgb": False,
    "depth_image": True,
}
Code = {
    "segmentation": False,
    "read_data": True,
    "obstacle_avoidance": True,
}
