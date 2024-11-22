import serial
from read_data_utils import read_serial_data, visualize2D
import configs.config as cfg
import cv2
import numpy as np
from utils.segmention import Two_Pass
import read_data_utils as read_data_utils
import time

def find_max(distances, res=8):
    max_dist = 1e-9
    max_i = 0
    max_j = 0
    for i in range(res):
        for j in range(res):
            if distances[i][j] == 0:
                continue
            elif distances[i][j] > max_dist:
                max_dist = distances[i][j]
                max_i = i
                max_j = j
    return max_i, max_j

def refine_by_time(last_distances, last_sigma, distances, sigma, res=8):
    refine_distance  = np.zeros_like(distances)
    refine_sigma = np.zeros_like(sigma)
    for i in range(res):
        for j in range(res):
            if sigma[i][j] == 0:
                refine_distance[i][j] = last_distances[i][j]
                refine_sigma[i][j] = last_sigma[i][j]
                continue
            elif last_sigma[i][j] == 0:
                refine_distance[i][j] = distances[i][j]
                refine_sigma[i][j] = sigma[i][j]
                continue
            else:
                refine_distance[i][j] = (last_distances[i][j] * sigma[i][j] + distances[i][j] * last_sigma[i][j]) \
                    / (sigma[i][j] + last_sigma[i][j])
                refine_sigma[i][j] = (sigma[i][j] * last_sigma[i][j]) / (sigma[i][j] + last_sigma[i][j])
    return refine_distance, refine_sigma

def sptial_filter(distances, sigma, kernel_size = 3, res=8):
    ## 3x3 filter with a sigma weight
    ## 8*8 array to 6*6 array
    result = np.zeros((res - kernel_size + 1, res - kernel_size + 1))
    for i in range(res - kernel_size + 1):
        for j in range(res - kernel_size + 1):
            result[i][j] = np.sum(distances[i:i+kernel_size, j:j+kernel_size] * 1 / sigma[i:i+kernel_size, j:j+kernel_size]) \
                / np.sum(1 / sigma[i:i+kernel_size, j:j+kernel_size])
    return result

def draw_arrow(img, direction):
    out_width, out_height = cfg.Sensor["output_shape"]
    match direction:
        case "up":
            cv2.arrowedLine(img, (out_width // 2, out_height // 2), (out_width // 2, 0), (0, 0, 255), 5)
        case "down":
            cv2.arrowedLine(img, (out_width // 2, out_height // 2), (out_width // 2, out_height), (0, 0, 255), 5)
        case "left":
            cv2.arrowedLine(img, (out_width // 2, out_height // 2), (0, out_height // 2), (0, 0, 255), 5)
        case "right":
            cv2.arrowedLine(img, (out_width // 2, out_height // 2), (out_width, out_height // 2), (0, 0, 255), 5)
        case "up-left":
            cv2.arrowedLine(img, (out_width // 2, out_height // 2), (0, 0), (0, 0, 255), 5)
        case "up-right":
            cv2.arrowedLine(img, (out_width // 2, out_height // 2), (out_width, 0), (0, 0, 255), 5)
        case "down-left":
            cv2.arrowedLine(img, (out_width // 2, out_height // 2), (0, out_height), (0, 0, 255), 5)
        case "down-right":
            cv2.arrowedLine(img, (out_width // 2, out_height // 2), (out_width, out_height), (0, 0, 255), 5)
        case _:
            x, y = direction   

def depth_only_visualize(max_row, max_column, color_depth, spatial_refine_distances):
    if max_row <= spatial_refine_distances.shape[0] // 2 - 2:
        if max_column <= spatial_refine_distances.shape[1] // 2 - 2:
            draw_arrow(color_depth, "up-left")
            cv2.imshow('depth', color_depth)
            cv2.waitKey(1) & 0xFF == ord('q')
        elif max_column >= spatial_refine_distances.shape[1] // 2 + 2:
            draw_arrow(color_depth, "up-right")
            cv2.imshow('depth', color_depth)
            cv2.waitKey(1) & 0xFF == ord('q')
        else:
            draw_arrow(color_depth, "up")
            cv2.imshow('depth', color_depth)
            cv2.waitKey(1) & 0xFF == ord('q')
    elif max_row >= spatial_refine_distances.shape[0] // 2 + 2:
        if max_column <= spatial_refine_distances.shape[1] // 2 - 2:
            draw_arrow(color_depth, "down-left")
            cv2.imshow('depth', color_depth)
            cv2.waitKey(1) & 0xFF == ord('q')
        elif max_column >= spatial_refine_distances.shape[1] // 2 + 2:
            draw_arrow(color_depth, "down-right")
            cv2.imshow('depth', color_depth)
            cv2.waitKey(1) & 0xFF == ord('q')
        else:
            draw_arrow(color_depth, "down")
            cv2.imshow('depth', color_depth)
            cv2.waitKey(1) & 0xFF == ord('q')

def depth_and_seg_visualize(max_row, max_column, color_depth, binary_img, spatial_refine_distances):
    if max_row <= spatial_refine_distances.shape[0] // 2 - 2:
        if max_column <= spatial_refine_distances.shape[1] // 2 - 2:
            draw_arrow(color_depth, "up-left")
            cv2.imshow('depth', np.hstack([color_depth, binary_img]))
            cv2.waitKey(1) & 0xFF == ord('q')
        elif max_column >= spatial_refine_distances.shape[1] // 2 + 2:
            draw_arrow(color_depth, "up-right")
            cv2.imshow('depth', np.hstack([color_depth, binary_img]))
            cv2.waitKey(1) & 0xFF == ord('q')
        else:
            draw_arrow(color_depth, "up")
            cv2.imshow('depth', np.hstack([color_depth, binary_img]))
            cv2.waitKey(1) & 0xFF == ord('q')
    elif max_row >= spatial_refine_distances.shape[0] // 2 + 2:
        if max_column <= spatial_refine_distances.shape[1] // 2 - 2:
            draw_arrow(color_depth, "down-left")
            cv2.imshow('depth', np.hstack([color_depth, binary_img]))
            cv2.waitKey(1) & 0xFF == ord('q')
        elif max_column >= spatial_refine_distances.shape[1] // 2 + 2:
            draw_arrow(color_depth, "down-right")
            cv2.imshow('depth', np.hstack([color_depth, binary_img]))
            cv2.waitKey(1) & 0xFF == ord('q')
        else:
            draw_arrow(color_depth, "down")
            cv2.imshow('depth', np.hstack([color_depth, binary_img]))
            cv2.waitKey(1) & 0xFF == ord('q')
        

if __name__ == "__main__":
    ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
    last_distances = np.zeros((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    last_sigma = np.ones((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    pad_size = cfg.Sensor["output_shape"][0] // cfg.Sensor["resolution"]
    NEIGHBORHOOD_8 = True
    while True:
        distances, sigma = read_serial_data(ser, cfg.Sensor["resolution"])
        ## 1. refine by time
        time_refine_distances, time_refine_sigma = refine_by_time(distances, sigma, last_distances, last_sigma)
        last_distances = distances
        last_sigma = sigma
        ## 2. spatial filter
        spatial_refine_distances = sptial_filter(time_refine_distances, time_refine_sigma)
        ## 3. find the max distance
        max_row, max_column = find_max(spatial_refine_distances, spatial_refine_distances.shape[0])
        ## 4. raw depth image and color depth image
        depth = visualize2D(time_refine_distances, time_refine_sigma, cfg.Sensor["resolution"], cfg.Sensor["output_shape"])
        color_depth = cv2.applyColorMap(read_data_utils.normalize(depth), cv2.COLORMAP_MAGMA)
        ## 5. segmention
        if cfg.Code["segmentation"]:
            binary_img = np.zeros_like(color_depth)
            var = Two_Pass(depth, NEIGHBORHOOD_8)
            binary_img = var[:,:,np.newaxis].repeat(3, axis=2)

        cv2.circle(color_depth, (round(max_column * pad_size * 4/3), round(max_row * pad_size * 4/3)), 10, (0, 255, 0), -1)
        cv2.imwrite(f'direction/{time.time()}.png', color_depth)
        cv2.imshow('depth', color_depth)
        if cfg.Code["segmentation"]:
            depth_and_seg_visualize(max_row, max_column, color_depth, binary_img, spatial_refine_distances)
        else:
            depth_only_visualize(max_row, max_column, color_depth, spatial_refine_distances)

        