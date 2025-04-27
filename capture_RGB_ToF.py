'''
This script captures RGB and ToF images from a camera and saves them as the format with ZJUL5.
-.h5 file
--depth: 640*480
--fr: sy,sx,ey,ex
--hist_data: 64*2 [ToF depth data, ToF variaty]
--mask: 64
--rgb: 640*480*3
'''

import os
import cv2
import numpy as np
import h5py
import time
import serial
from read_data_utils import read_serial_data
import configs.config as cfg



def save_h5(data, file_name, h5_cfg):
    with h5py.File(file_name, "w") as f:
        if h5_cfg["hist_data"]:
            f.create_dataset("hist_data", data=data["hist_data"])
        if h5_cfg["depth"]:
            f.create_dataset("depth", data=data["depth"])
        if h5_cfg["rgb"]:
            f.create_dataset("rgb", data=data["rgb"])
        if h5_cfg["fr"]:
            f.create_dataset("fr", data=data["fr"])
        if h5_cfg["mask"]:
            f.create_dataset("mask", data=data["mask"])



def capture_rgb_tof_data(output_dir):
    ## Assume the FoV of ToF and RGB camera are the centralized. 640*480 -> 8*8
    fr = np.zeros((64, 4), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            fr[i * 8 + j, 0] = i * 60
            fr[i * 8 + j, 1] = j * 60 + 80
            fr[i * 8 + j, 2] = (i+1) * 60
            fr[i * 8 + j, 3] = (j+1) * 60 + 80

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(1)
    # 检查摄像头是否打开成功
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
<<<<<<< HEAD
    interval = 0.2  # seconds
=======
    interval = 0.02  # seconds
>>>>>>> 16d7ecdd60a63e5ad33dc48c3bff30d116fc2fac
    last_time = time.time()
    
    # Read RGB data from the camera
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取视频帧")
                break

            # 当前时间
            current_time = time.time()
            if current_time - last_time >= interval:
                # Read data from serial port
                distances, sigma, mask = read_serial_data(ser, cfg.Sensor["resolution"])
                hist_data = np.zeros((64, 2), dtype=np.float32)
                hist_data[:, 0] = distances.flatten()
                hist_data[:, 1] = sigma.flatten()
                mask = mask.flatten()
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                date = time.strftime("%Y-%m-%d")
                if not os.path.exists(os.path.join(output_dir, date)):
                    os.makedirs(os.path.join(output_dir, date))
                # Save data to .h5 file 
                h5_file_name = os.path.join(output_dir, date, f"{timestamp}.h5")
                save_h5({"hist_data": hist_data, "fr": fr, "mask": mask, "rgb":frame}, h5_file_name, cfg.h5_cfg)
                print(f"save data: {h5_file_name}")
                last_time = current_time
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        ser.close()


if __name__ == "__main__":
    output_dir = "data"
    capture_rgb_tof_data(output_dir)