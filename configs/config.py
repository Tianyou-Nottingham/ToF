Serial = {
    "port": "COM5",
    "baudrate": 460800,
}
Sensor = {
    "resolution": 8,
<<<<<<< HEAD
    "max_depth": 2000, ## 10 meters
=======
    "max_depth": 2500, ## milemeters
>>>>>>> 158830bf15fffd7c5d28d6c483d2cd29100af011
    "min_depth": 1e-3,
    "output_shape": [640, 640],
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