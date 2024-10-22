import numpy as np
import configs.config as cfg
import serial
from read_data import read_serial_data, visualize2D
import cv2
import read_data
from sklearn.inspection import DecisionBoundaryDisplay


## For ToF sensor, we think the imaging model is the same as camera: a pinhole camera model.
## Only difference is the resolution.
## For camera and ToF calibration, we can use the same calibration method.
## Detect the corners of the chessboard, and use the corners to calibrate the camera.
## But how to detect the corners of the chessboard in ToF data?
## 1. We need a different-depth chessboard.
## 2. We need to detect the corners in the low-resolution depth image.
## So this code is to detect the corners in the low-resolution depth image.
## Of course, we need a video input or continuous depth image input.

class Line:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.k = (end[1] - start[1]) / (end[0] - start[0])
        self.b = start[1] - self.k * start[0]

    def __str__(self):
        return f"Line: {self.start} -> {self.end}"

def padding(data, pad_size):
    w, h = data.shape
    pad_data = np.zeros((w + 2 * pad_size, h + 2 * pad_size))
    pad_data[pad_size: w + pad_size, pad_size: h + pad_size] = data
    return pad_data


def tof_to_camera(x, y, image_shape):
    ## transform the ToF image to camera image
    pad_size = image_shape // cfg.Sensor["resolution"]
    return [(x + 1/2) * pad_size, (y + 1/2) * pad_size]

def kmeans_clustering(data, k):
    ## data:[8, 8] depth map
    ## k: number of clusters
    ## return the segmentation line
    ## For the zones divided by the segmentation line, we calculate the mean depth value. And set the new 
    w, h = data.shape
    centers = [np.random.choice(w, k), np.random.choice(h, k)]
    cluster_index = [[] for _ in range(k)]
    cluster_value = [[] for _ in range(k)]
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(w):
            for j in range(h):
                min_dist = np.inf
                min_index = -1
                dist = []

                for x, y in centers:
                    dist.append((data[i, j] - data[round(x), round(y)])**2)
                min_index = np.argmin(dist)
                cluster_value[min_index].append(data[i, j])
                if (i, j) not in cluster_index[min_index]:
                    cluster_changed = True
                if min_dist > dist[min_index]:
                    min_dist = dist[min_index]
                    cluster_index[min_index].append((i, j))
        value = {}
        for idx in range(k):
            centers[idx] = np.mean(cluster_index[idx], axis=0)
            value.update({np.mean(cluster_value[idx], axis=0): centers[idx]})
        
        value = sorted(value.items(), key=lambda x: x[0])
    return list(dict(value).values())

def edge_detect(data):
    padding_data = padding(data, 1)
    vertical_edge = np.zeros_like(data)
    horizontal_edge = np.zeros_like(data)
    for i in range(1, data.shape[0] + 1):
        for j in range(1, data.shape[1] + 1):
            ## Sobel operator
            vertical_edge[i - 1, j - 1] = (padding_data[i - 1, j + 1] + 2 * padding_data[i, j + 1] + padding_data[i + 1, j + 1] \
                - padding_data[i - 1, j - 1] - 2 * padding_data[i, j - 1] - padding_data[i + 1, j - 1]) / 6
            horizontal_edge[i - 1, j - 1] = (padding_data[i + 1, j - 1] + 2 * padding_data[i + 1, j] + padding_data[i + 1, j + 1] \
                - padding_data[i - 1, j - 1] - 2 * padding_data[i - 1, j] - padding_data[i - 1, j + 1]) / 6
    return vertical_edge, horizontal_edge

def line_detect(data, threshold):
    vertical_edge, horizontal_edge = edge_detect(data)
    vertical_edge = np.abs(vertical_edge)
    horizontal_edge = np.abs(horizontal_edge)
    verticle_edge_center = []
    horizontal_edge_center = []
    for i in range(vertical_edge.shape[0]):
        for j in range(vertical_edge.shape[1]):
            if data[i, j] == 0:
                continue
            elif vertical_edge[i, j] > threshold:
                verticle_edge_center.append((i, j))
            elif horizontal_edge[i, j] > threshold:
                horizontal_edge_center.append((i, j))
    if len(verticle_edge_center) == 0 or len(horizontal_edge_center) == 0:
        return None, None
    else:
        vertical_lines = np.polyfit(verticle_edge_center[0], verticle_edge_center[1], 1)
        horizontal_lines = np.polyfit(horizontal_edge_center[0], horizontal_edge_center[1], 1)
    return vertical_lines, horizontal_lines


def main():
    ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
    pad_size = cfg.Sensor["output_shape"][0] // cfg.Sensor["resolution"]
    output_shape = cfg.Sensor["output_shape"]
    while True:
        ## 1. Read the data from the serial port
        distances, sigma = read_serial_data(ser, cfg.Sensor["resolution"])
        ## 2. K-means clustering
        centers = kmeans_clustering(distances, 2)
        depth = visualize2D(distances, sigma, cfg.Sensor["resolution"], cfg.Sensor["output_shape"])

        # vertical_edge, horizontal_edge = edge_detect(distances)
        # vertical_edge = visualize2D(vertical_edge, sigma, cfg.Sensor["resolution"], cfg.Sensor["output_shape"])
        # horizontal_edge = visualize2D(horizontal_edge, sigma, cfg.Sensor["resolution"], cfg.Sensor["output_shape"])
        # vertical_edge = cv2.applyColorMap(read_data.normalize(vertical_edge), cv2.COLORMAP_MAGMA)
        # horizontal_edge = cv2.applyColorMap(read_data.normalize(horizontal_edge), cv2.COLORMAP_MAGMA)
        # cv2.imshow('vertical_edge', vertical_edge)
        # cv2.imshow('horizontal_edge', horizontal_edge)
        # vertical_line, horizontal_line = line_detect(distances, 500)

        color_depth = cv2.applyColorMap(read_data.normalize(depth), cv2.COLORMAP_MAGMA)        
        cv2.circle(color_depth, (round(centers[0][1]*pad_size), round(centers[0][0]*pad_size)), 5, (0, 255, 0), -1)
        cv2.putText(color_depth, "Obstacle", (round(centers[0][1]*pad_size), round(centers[0][0]*pad_size)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(color_depth, (round(centers[1][1]*pad_size), round(centers[1][0]*pad_size)), 5, (0, 0, 255), -1)
        cv2.putText(color_depth, "Safe", (round(centers[1][1]*pad_size), round(centers[1][0]*pad_size)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # if vertical_line is not None :
        #     cv2.line(color_depth, (0, round(vertical_line[1] * pad_size)), (output_shape[0], round(vertical_line[0] * output_shape[0] + vertical_line[1] * pad_size)), (0, 255, 0), 2)
        # if horizontal_line is not None:
        #     cv2.line(color_depth, (round(horizontal_line[1] * pad_size), 0), (round(horizontal_line[0] * output_shape[0] + horizontal_line[1] * pad_size), output_shape[0]), (0, 0, 255), 2)
        cv2.imshow('depth', color_depth)
        cv2.waitKey(1) & 0xFF == ord('q')

        
if __name__ == "__main__":
    main() 