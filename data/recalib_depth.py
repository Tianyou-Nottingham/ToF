import  os
import cv2
import numpy as np
import h5py

## read data from .h5 file and rewrite depth by times depth scale
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

h5_dir = "E:/Projects/ToF/ToF/data/2025-07-16"
depth_scale = 9.999999747378752e-05

if __name__ == "__main__":
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    for h5_file in h5_files:
        file_path = os.path.join(h5_dir, h5_file)
        with h5py.File(file_path, 'r+') as f:
            if 'depth' in f:
                depth_data = f['depth'][:]
                # Scale the depth data
                scaled_depth_data = depth_data * depth_scale
                # Update the dataset
                f['depth'][:] = scaled_depth_data
                print(f"Updated depth data in {h5_file}")
            else:
                print(f"No depth data found in {h5_file}")