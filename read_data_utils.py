import cv2
import numpy as np
import serial
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import h5py
matplotlib.use('TkAgg',force=True)
from scipy import interpolate
import configs.config as cfg
# PythonDataStart
# |    63  :      5 |    66  :      5 |    71  :      5 |    73  :      5 |    74  :      5 |    79  :      5 |    83  :      5 |     X  :      X |
# |    66  :      5 |    69  :      5 |    74  :      5 |    74  :      5 |    76  :      5 |    81  :      5 |     X  :      X |     X  :      X |
# |    70  :      5 |    71  :      5 |    76  :      5 |    76  :      5 |    79  :      5 |    84  :      5 |    92  :      5 |    96  :      5 |
# |    73  :      5 |    74  :      5 |    78  :      5 |    79  :      5 |    85  :      5 |    91  :      5 |    92  :      5 |   100  :      5 |
# |    73  :      5 |    78  :      5 |    81  :      5 |    85  :      5 |    87  :      5 |     X  :      X |    96  :      5 |   105  :      5 |
# |    78  :      5 |    80  :      5 |    86  :      5 |    87  :      5 |    92  :      5 |     X  :      X |   102  :      5 |   100  :      5 |
# |    83  :      5 |    86  :      5 |    91  :      5 |    91  :      5 |     X  :      X |    99  :      5 |   104  :      5 |    95  :      5 |
# |    84  :      5 |    91  :      5 |    96  :      5 |    98  :      5 |   103  :      5 |    98  :     10 |    99  :      5 |    82  :      5 |
def normalize(value, vmin=cfg.Sensor["min_depth"], vmax=cfg.Sensor["max_depth"]):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    value = (value*255.0).astype(np.uint8)
    return value

def parse_data(raw_data):
    distance = []
    sigma = []
    
    for line in raw_data:
        row_first = []
        row_second = []
        pairs = line.split(b'|')[1:-1]  # Split by '|' and ignore the first and last empty strings
        for pair in pairs:
            first, second = pair.split(b':')
            if first.strip() == b'X':
                row_first.append(1)
                row_second.append(1)
                continue
            row_first.append(int(first.strip()))
            row_second.append(int(second.strip()))
        distance.append(row_first)
        sigma.append(row_second)
    
    distance.reverse()
    sigma.reverse()
    return np.array(distance), np.array(sigma)

def read_serial_data(ser, res=8):
    if ser.readline().strip() == b'PythonDataStart':
        raw_data = [ser.readline().strip() for _ in range(res)]
    distances, sigma = parse_data(raw_data)
    return distances, sigma ## 8x8 array
    
def visualize2D(distances, sigma, res=8, output_shape=[640, 640], upsample=False):
        # print(distances)
        depth_map = np.zeros(output_shape).astype(np.uint8)
        sigma_map = np.zeros(output_shape).astype(np.uint8)
        out_width, out_height = output_shape
        pad_size = out_width // res
        distances = distances.astype(np.uint8)
        sigma = sigma.astype(np.uint8)
        if upsample:
            depth_map = cv2.resize(distances, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
            sigma_map = cv2.resize(sigma, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
        else:
            for i in range(res):
                for j in range(res):
                    depth_map[i*pad_size:(i+1)*pad_size, j*pad_size:(j+1)*pad_size] = distances[i, j]
                    sigma_map[i*pad_size:(i+1)*pad_size, j*pad_size:(j+1)*pad_size] = sigma[i, j]
        # depth = cv2.applyColorMap(normalize(depth), cv2.COLORMAP_MAGMA)
        # cv2.imshow('depth', depth)
        # img_name = f'output/{time.time()}.png'
        # data['depth_image'] = depth
        # cv2.imwrite(img_name, depth)
        # cv2.waitKey(1) & 0xFF == ord('q')
        return depth_map, sigma_map

def read_file_data(file_name, res):
    with open(file_name, 'r') as f:
        while f.readline().strip() != 'PythonDataStart':
            pass
        # read 8 lines of data
        data = [f.readline().strip() for _ in range(res)]
    return data

def save_h5(data, file_name, h5_cfg):
    with h5py.File(file_name, 'w') as f:
        if h5_cfg["distance"]:
            f.create_dataset('distance', data=data["distance"])
        if h5_cfg["sigma"]:
            f.create_dataset('sigma', data=data["sigma"])
        if h5_cfg["rgb"]:
            f.create_dataset('rgb', data=data["rgb"])
        if h5_cfg["depth_image"]:
            f.create_dataset('depth_image', data=data["depth_image"])

if __name__ == "__main__":
    ######### Define the source of the data! #########
    source = 'serial'  # 'file' or 'serial'

    ######### File Configuration #########
    file_name = 'CoolTerm Capture 2024-10-02 11-37-02.txt'

    data = {}

    if source == 'file':
        raw_data = read_file_data(file_name, cfg.Sensor["resolution"])

    else:
        ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
        while True:
            distances, sigma = read_serial_data(ser, cfg.Sensor["resolution"])
<<<<<<< HEAD:read_data.py
            distances = normalize(distances)
=======
            # distances = normalize(distances)
>>>>>>> 158830bf15fffd7c5d28d6c483d2cd29100af011:read_data_utils.py
            print(distances)
            depth_map, sigma_map = visualize2D(distances, sigma, cfg.Sensor["resolution"], cfg.Sensor["output_shape"], upsample=False)
            data['distance'] = distances
            data['sigma'] = sigma
            color_depth = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)        
            cv2.imshow('depth', color_depth)
            cv2.waitKey(1) & 0xFF == ord('q')
<<<<<<< HEAD:read_data.py

            # h5_name = f'output/{time.time()}.h5'
            # save_h5(data, h5_name, cfg.h5_cfg)
=======
>>>>>>> 158830bf15fffd7c5d28d6c483d2cd29100af011:read_data_utils.py


        # ser.close()
            