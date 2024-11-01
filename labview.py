from socket import *
import numpy as np
import configs.config as cfg
import serial
from read_data import read_serial_data, visualize2D
import cv2
import read_data
import time
from obstacle_avoidance import kmeans_clustering, Plane, outliers_detection
from direction_visualization import refine_by_time

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
    if center_x < output_shape // 2:
        if center_y < output_shape // 2:
            move_step = np.array([-5, -5])
        elif center_y > output_shape // 2:
            move_step = np.array([-5, 5])
        # else:
        #     move_step = np.array([-5, 0])
    elif center_x > output_shape // 2:
        if center_y < output_shape // 2:
            move_step = np.array([5, -5])
        elif center_y > output_shape // 2:
            move_step = np.array([5, 5])
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

        ## Cheange [x, x] data to TCP format
        direction = direction_of_safe_zone(center_safe, cfg.Sensor["resolution"])
        index += 1
        if index % 5 != 0:
            send = send + direction
        else:

            send = str(round(send[0]/5)).rjust(3, ' ')+","+str(round(send[1])).rjust(3, ' ')
        
            print('Send data: ', send.encode('utf-8'))
            print('Send data length: ', len(send))

            if TCP_R_data == b'OK':
                print('Received OK')
                TCP_send(connection_socket, send.encode('utf-8'))
                time.sleep(0.2)
            else:
                print('No data received')

            send = np.array([0, 0])

        depth, sigma = visualize2D(time_refine_distances, sigma, cfg.Sensor["resolution"], cfg.Sensor["output_shape"])
        color_depth = cv2.applyColorMap(read_data.normalize(depth), cv2.COLORMAP_MAGMA)        
        cv2.circle(color_depth, (round(center_obstacle[1]*pad_size), round(center_obstacle[0]*pad_size)), 5, (0, 255, 0), -1)
        cv2.putText(color_depth, "Obstacle", (round(center_obstacle[1]*pad_size), round(center_obstacle[0]*pad_size)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(color_depth, (round(center_safe[1]*pad_size), round(center_safe[0]*pad_size)), 5, (0, 0, 255), -1)
        cv2.putText(color_depth, "Safe", (round(center_safe[1]*pad_size), round(center_safe[0]*pad_size)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imwrite(f'kmeans/{time.time()}.png', color_depth)
        cv2.imshow('depth', color_depth)
        cv2.waitKey(1) & 0xFF == ord('q')

    exit(connection_socket)
    server_socket.close()

if __name__ == "__main__":
    main()
    