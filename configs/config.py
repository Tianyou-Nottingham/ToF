Serial = {
    "port": "COM5",
    "baudrate": 460800,
}
Sensor = {
    "resolution": 8,
    "max_depth": 2000, ## 10 meters
    "min_depth": 1e-3,
    "output_shape": [256, 256],
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