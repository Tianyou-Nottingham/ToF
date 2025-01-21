import numpy as np
import configs.config as cfg
import serial
from TOF_RANSAC import Plane
from read_data_utils import read_serial_data, visualize2D, normalize
from direction_visualization import refine_by_time
from obstacle_avoidance import kmeans_clustering, outliers_detection
import matplotlib.pyplot as plt
from utils.kmeans import plane_kmeans
from utils.distance_rectified_fov import distance_rectified_fov


def two_plane_visualization(fig, Plane1, Plane2, data1, data2):
    """
    两个平面可视化
    :param N1: 平面1的法向量(3维,形如 [a, b, c])
    :param d1: 平面1基点偏移量(3维向量,形如 [d_x, d_y, d_z])
    :param N2: 平面2的法向量(3维,形如 [a, b, c])
    :param d2: 平面2基点偏移量(3维向量,形如 [d_x, d_y, d_z])
    平面方程: N · X + d = 0 或 ax + by + cz + d_offset = 0
    """
    # 解析法向量
    a1, b1, c1 = Plane1.N
    a2, b2, c2 = Plane2.N

    # # 计算偏移量 d_offset
    # d_offset = -(a * d[0] + b * d[1] + c * d[2])

    # 创建网格 (x, y)
    x = np.linspace(-50, 50, 100)
    y = np.linspace(-50, 50, 100)
    X, Y = np.meshgrid(x, y)

    # 根据平面方程 N · X + d = 0 求解 z
    if c1 == 0 or c2 == 0:
        raise ValueError(
            "The normal vector's z component (c) cannot be zero for visualization."
        )
    Z1 = -(a1 * X + b1 * Y - Plane1.d) / c1
    Z2 = -(a2 * X + b2 * Y - Plane2.d) / c2

    # 创建 3D 图形
    ax = fig.add_subplot(121, projection="3d")

    # 绘制平面
    # if Z < 0:
    #     ax.plot_surface(-X, -Y, -Z, alpha=0.5, color='blue', edgecolor='k')
    # else:
    ax.plot_surface(X, Y, Z1, alpha=0.2, color="red")
    ax.plot_surface(X, Y, Z2, alpha=0.2, color="green")
    # 绘制数据点
    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c="r", marker="o")
    ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c="g", marker="o")
    ax.view_init(elev=50, azim=0)
    # 设置轴标签
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    # 设置标题
    ax.set_title("Plane Visualization")


def main():
    ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
    last_distances = np.zeros((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    last_sigma = np.ones((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    pad_size = cfg.Sensor["output_shape"][0] // cfg.Sensor["resolution"]
    output_shape = cfg.Sensor["output_shape"]
    points3D = []
    while True:
        ## 1. Read the data from the serial port
        distances, sigma = read_serial_data(ser, cfg.Sensor["resolution"])
        points3D = np.array(
            [
                [i, j, distances[i, j]]
                for i in range(cfg.Sensor["resolution"])
                for j in range(cfg.Sensor["resolution"])
            ]
        )
        ## 2. Refine by time
        time_refine_distances, time_refine_sigma = refine_by_time(
            distances, sigma, last_distances, last_sigma
        )
        last_distances = distances
        last_sigma = sigma
        ## 3. K-means clustering
        points_index = plane_kmeans(time_refine_distances, 2)
        ## 4. Outliers detection
        # points_index = outliers_detection(points_index, 4)
        ## 5. Plane fitting
        points_obstacle = np.array(
            [[i, j, time_refine_distances[i, j]] for [i, j] in points_index[0]]
        )
        points_safe = np.array(
            [[i, j, time_refine_distances[i, j]] for [i, j] in points_index[1]]
        )
        plane_obstacle = Plane(np.array([0, 1, 1]), 0)
        plane_safe = Plane(np.array([1, 0, 1]), 0)
        plane_obstacle, error_obstacle = plane_obstacle.ToF_RANSAC(
            points_obstacle, res=cfg.Sensor["resolution"]
        )
        plane_safe, error_safe = plane_safe.ToF_RANSAC(
            points_safe, res=cfg.Sensor["resolution"]
        )
        centers = [np.mean(points_index[0], axis=0), np.mean(points_index[1], axis=0)]
        ## 6. Visualization
        fig = plt.figure(figsize=(14, 7))
        # obstacle point and plane is red, safe point and plane is green
        two_plane_visualization(
            fig, plane_obstacle, plane_safe, points_obstacle, points_safe
        )
        plane_obstacle.ToF_visualization(
            fig,
            time_refine_distances,
            time_refine_sigma,
            cfg.Sensor["resolution"],
            cfg.Sensor["output_shape"],
        )

        # plane_obstacle.plane_visualization(fig, plane_obstacle.N, plane_obstacle.d, points_obstacle, color='r')
        # plane_safe.plane_visualization(fig, plane_safe.N, plane_safe.d, points_safe, color='g')
        # plane_obstacle.ToF_visualization(fig, time_refine_distances, time_refine_sigma, cfg.Sensor["resolution"], cfg.Sensor["output_shape"])
        # plane_safe.ToF_visualization(fig, time_refine_distances, time_refine_sigma, cfg.Sensor["resolution"], cfg.Sensor["output_shape"])
        plt.show()
        print(
            f"Obstacle plane N: {plane_obstacle.N}, d: {plane_obstacle.d}, Error: {error_obstacle}."
        )
        print(f"Safe plane N: {plane_safe.N}, d: {plane_safe.d}, Error: {error_safe}.")


def test():
    ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
    last_distances = np.zeros((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    last_sigma = np.ones((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
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
        grad_x = np.gradient(time_refine_distances, axis=1)
        grad_y = np.gradient(time_refine_distances, axis=0)

        ## 4. Divide the plane into two parts based on the angle
        plane1_index = []
        plane2_index = []
        for i in range(cfg.Sensor["resolution"]):
            for j in range(cfg.Sensor["resolution"]):
                if grad_x[i][j] > 0:
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
        plane1 = Plane(np.array([0, 0, 1]), 0)
        plane2 = Plane(np.array([0, 0, 1]), 0)
        # plane1.fit_plane(points_plane1)
        # plane2.fit_plane(points_plane2)
        # ToF_RANSAC will transfer the points to the world coordinate
        plane1 = plane1.ToF_RANSAC(points_plane1, res=cfg.Sensor["resolution"])
        plane2 = plane2.ToF_RANSAC(points_plane2, res=cfg.Sensor["resolution"])
        ## 6. Visualization
        fig = plt.figure(figsize=(14, 7))
        # Transfer the visulaization to the world coordinate
        if cfg.Code["distance_rectified_fov"]:
            points1 = distance_rectified_fov(points_plane1)
            points2 = distance_rectified_fov(points_plane2)
        two_plane_visualization(fig, plane1, plane2, points1, points2)
        plane1.ToF_visualization(
            fig,
            time_refine_distances,
            time_refine_sigma,
            cfg.Sensor["resolution"],
            cfg.Sensor["output_shape"],
        )
        plt.show()
        print(f"Plane1 N: {plane1.N}, d: {plane1.d}, Error: {plane1.error}.")
        print(f"Plane2 N: {plane2.N}, d: {plane2.d}, Error: {plane2.error}.")


if __name__ == "__main__":
    test()
