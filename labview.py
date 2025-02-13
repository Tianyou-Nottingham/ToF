from socket import *
import numpy as np
import configs.config as cfg
import serial
from read_data_utils import read_serial_data, visualize2D
import cv2
import read_data_utils
import time
from obstacle_avoidance import kmeans_clustering, Plane, outliers_detection
from direction_visualization import refine_by_time
from utils.distance_rectified_fov import distance_rectified_fov
from two_plane_fit import two_planes_fitting, two_plane_visualization
import matplotlib.pyplot as plt


def TCP_server_connect():
    server_socket = socket(AF_INET, SOCK_STREAM)

    server_socket.bind(('127.0.0.1', 8088))
    server_socket.listen(1)

    print('Waiting for connection...')
    connection_socket, addr = server_socket.accept()
    print('Connected by', addr)
    return connection_socket

def TCP_send(socket, data):
    socket.send(data)

def TCP_receive(socket):
    data = socket.recv(32)
    return data

def exit(socket):
    print('Connection closed')
    socket.close()

def direction_of_safe_zone(center_safe, output_shape):
    center_x = center_safe[0]
    center_y = center_safe[1]
    move_step = np.array([0, 0])
    # Robot's coordinate system is different from the image coordinate system
    # x-axis is opposite
    if center_x < output_shape // 2: # safe center is on the image up side, robot should move up
        if center_y < output_shape // 2: # safe center is on the image left side, robot should move up and left
            move_step = np.array([-1, 1])
        elif center_y > output_shape // 2:
            move_step = np.array([1, 1])
        # else:
        #     move_step = np.array([-5, 0])
    elif center_x > output_shape // 2:
        if center_y < output_shape // 2:
            move_step = np.array([-1, -1])
        elif center_y > output_shape // 2:
            move_step = np.array([1, -1])
        # else:
        #     move_step = np.array([5, 0])
    # else:
    #     move_step = np.array([0, 0])
    return move_step

def main():
    ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
    last_distances = np.zeros((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    last_sigma = np.ones((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    pad_size = cfg.Sensor["output_shape"][0] // cfg.Sensor["resolution"]
    output_shape = cfg.Sensor["output_shape"]
    points3D = []
    try:
        connection_socket = TCP_server_connect()
        print('Connected')
        ## Send the data to the TCP server
        TCP_R_data = TCP_receive(connection_socket)
        print('Received data: ', TCP_R_data)
    except Exception as e:
        print(e)
        return

    index = 1
    send = np.array([0, 0])
    while True:
        ## 1. Read the data from the serial port
        distances, sigma = read_serial_data(ser, cfg.Sensor["resolution"])
        points3D = np.array([[i, j, distances[i, j]] for i in range(cfg.Sensor["resolution"]) for j in range(cfg.Sensor["resolution"])])
        ## 2. Refine by time
        time_refine_distances, time_refine_sigma = refine_by_time(distances, sigma, last_distances, last_sigma)
        last_distances = distances
        last_sigma = sigma
        ## 3. K-means clustering
        points_index = kmeans_clustering(time_refine_distances, 2)
        ## 4. Outliers detection
        points_index = outliers_detection(points_index, 4)
        points_obstacle = np.array([[i, j, time_refine_distances[i, j]] for [i, j] in points_index[0]])
        points_safe = np.array([[i, j, time_refine_distances[i, j]] for [i, j] in points_index[1]])
        center_obstacle = np.mean(points_index[0], axis=0)
        center_safe = np.mean(points_index[1], axis=0)
        # if np.linalg.norm(center_obstacle - center_safe, 2) < 2:
        #     continue
        ## Cheange [x, x] data to TCP format
        print('Obstacle center: ', center_obstacle)
        print('Safe center: ', center_safe)
        direction = direction_of_safe_zone(center_safe, cfg.Sensor["resolution"])

        send = str(direction[0]).rjust(4, ' ')+","+str(direction[1]).rjust(4, ' ')
        
        print('Send data: ', send.encode('utf-8'))
        print('Send data length: ', len(send))

        if TCP_R_data == b'OK':
            print('Received OK')
            TCP_send(connection_socket, send.encode('utf-8'))
        else:
            print('No data received')

        send = np.array([0, 0])

        depth, sigma = visualize2D(time_refine_distances, time_refine_sigma, cfg.Sensor["resolution"], cfg.Sensor["output_shape"])
        color_depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)        
        cv2.circle(color_depth, (round(center_obstacle[1]*pad_size), round(center_obstacle[0]*pad_size)), 5, (0, 255, 0), -1)
        cv2.putText(color_depth, "Obstacle", (round(center_obstacle[1]*pad_size), round(center_obstacle[0]*pad_size)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(color_depth, (round(center_safe[1]*pad_size), round(center_safe[0]*pad_size)), 5, (0, 0, 255), -1)
        cv2.putText(color_depth, "Safe", (round(center_safe[1]*pad_size), round(center_safe[0]*pad_size)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imwrite(f'kmeans/{time.time()}.png', color_depth)
        cv2.imshow('depth', color_depth)
        cv2.waitKey(1) & 0xFF == ord('q')

    exit(connection_socket)
    server_socket.close()

def twoPlaneDEMO():
    ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
    last_distances = np.zeros((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    last_sigma = np.ones((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    pad_size = cfg.Sensor["output_shape"][0] // cfg.Sensor["resolution"]
    output_shape = cfg.Sensor["output_shape"]
    points3D = []
    try:
        connection_socket = TCP_server_connect()
        print('Connected')
        ## Send the data to the TCP server
        TCP_R_data = TCP_receive(connection_socket)
        print('Received data: ', TCP_R_data)
    except Exception as e:
        print(e)
        return

    index = 1
    send = np.array([0, 0])
    while True:
        ## 1. Read the data from the serial port
        distances, sigma = read_serial_data(ser, cfg.Sensor["resolution"])
        ## 2. Refine by time
        time_refine_distances, time_refine_sigma = refine_by_time(
            distances, sigma, last_distances, last_sigma
        )
        last_distances = distances
        last_sigma = sigma

        points3D = np.array(
            [
                [i, j, time_refine_distances[i, j]]
                for i in range(cfg.Sensor["resolution"])
                for j in range(cfg.Sensor["resolution"])
            ]
        )

        # if cfg.Code["distance_rectified_fov"]:
        #     points_world = distance_rectified_fov(points3D)

        ## 3. Calculate the gradioents of x and y direction
        grad_y = np.gradient(time_refine_distances, axis=1)
        grad_x = np.gradient(time_refine_distances, axis=0)

        ## 4. Divide the plane into two parts based on the angle
        plane1_index = []
        plane2_index = []
        for i in range(cfg.Sensor["resolution"]):
            for j in range(cfg.Sensor["resolution"]):
                if grad_y[i][j] > 0:
                    plane1_index.append([i, j])
                else:
                    plane2_index.append([i, j])
        plane1_index = np.array(plane1_index)
        plane2_index = np.array(plane2_index)
        ## 5. Outliers detection
        plane1_index = outliers_detection(plane1_index, 4)
        plane2_index = outliers_detection(plane2_index, 4)
        points_plane1 = np.array([points3D[i * 8 + j] for [i, j] in plane1_index])
        points_plane2 = np.array([points3D[i * 8 + j] for [i, j] in plane2_index])
        ## 5. Plane fitting
        if cfg.Code["distance_rectified_fov"]:
            points_plane1 = distance_rectified_fov(np.array(points_plane1))
            points_plane2 = distance_rectified_fov(np.array(points_plane2))
        plane1 = Plane(np.array([0, 0, 1]), 0)
        plane2 = Plane(np.array([0, 0, 1]), 0)
        plane1, plane2 = two_planes_fitting(points_plane1, points_plane2)
        # ToF_RANSAC will transfer the points to the world coordinate
        # plane1 = plane1.ToF_RANSAC(points_plane1, res=cfg.Sensor["resolution"])
        # plane2 = plane2.ToF_RANSAC(points_plane2, res=cfg.Sensor["resolution"])
        ## 6. Visualization
        # fig = plt.figure(figsize=(14, 7))
        # Transfer the visulaization to the world coordinate
        # two_plane_visualization(fig, plane1, plane2, points_plane1, points_plane2)
        # plane1.ToF_visualization(
        #     fig,
        #     time_refine_distances,
        #     time_refine_sigma,
        #     cfg.Sensor["resolution"],
        #     cfg.Sensor["output_shape"],
        # )
        # plt.show()
        print(f"Plane1 N: {plane1.N}, d: {plane1.d}, Error: {plane1.error}.")
        print(f"Plane2 N: {plane2.N}, d: {plane2.d}, Error: {plane2.error}.")

        ## Try gradient for direction at first.
        if abs(grad_y.max()) > abs(grad_y.min()):
            safe_points = points_plane1
        else:
            safe_points = points_plane2
        center_safe = np.mean(safe_points, axis=0)
        direction = direction_of_safe_zone(center_safe, cfg.Sensor["resolution"])

        send = str(direction[0]).rjust(4, ' ')+","+str(direction[1]).rjust(4, ' ')
        
        print('Send data: ', send.encode('utf-8'))
        print('Send data length: ', len(send))

        if TCP_R_data == b'OK':
            print('Received OK')
            TCP_send(connection_socket, send.encode('utf-8'))
        else:
            print('No data received')

        send = np.array([0, 0])

        depth, sigma = visualize2D(time_refine_distances, time_refine_sigma, cfg.Sensor["resolution"], cfg.Sensor["output_shape"])
        color_depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)        
        cv2.circle(color_depth, (round(center_safe[1]*pad_size), round(center_safe[0]*pad_size)), 5, (0, 0, 255), -1)
        cv2.putText(color_depth, "Safe", (round(center_safe[1]*pad_size), round(center_safe[0]*pad_size)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imwrite(f'kmeans/{time.time()}.png', color_depth)
        cv2.imshow('depth', color_depth)
        cv2.waitKey(1) & 0xFF == ord('q')

    exit(connection_socket)
    server_socket.close()
if __name__ == "__main__":
    twoPlaneDEMO()
    