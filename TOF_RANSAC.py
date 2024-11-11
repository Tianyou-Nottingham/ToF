'''
This file is for ToF data plane fitting using RANSAC algorithm.
1. Read the data from the serial port or file.
2. Choose the resolution of the data. (Because the raw data is in 8x8 zones, but there must be a point in the zone of which value 
    is the zone's value. For example, if the zones is 8x8, we choose a refine resolution of 256*256. It means each zone has 32*32
    points. And there is a point's depth is just equal to the zone's value.)
3. For every zones, random chose a point. And randomly choose at least 3 zones to fit a plane.
4. Don't take other points in the same zone into account. Just take the chosed one point in the zone to apply RANSAC.
5. RANSAC algorithm
6. Choose other points in the same zone to apply RANSAC.
'''
from read_data import read_serial_data
import numpy as np
import cv2
import time

class Plane:
    def __init__(self, N, d) -> None:
        self.N = N
        self.d = d

    def solve_distance(self, point):
        """
        平面方程: N * x + d = 0
        """
        return np.dot(self.N, point) + self.d
    
    def solve_plane(self, A, B, C):
        """
        求解平面方程
        :params: three points
        :return: Nx(平面法向量), d
        """
        plane = Plane(np.array([0, 0, 1]), 0)
        Nx = np.cross(B - A, C - A)
        Nx = Nx / np.linalg.norm(Nx)
        d = -np.dot(Nx, np.mean(A+B+C))
        plane.N = Nx
        plane.d = d
        return plane
    
    def RANSAC(self, data):
        """
        RANSAC算法
        :params: data: 3D points
        :return: best_plane: 最优平面
        """
        best_plane = None
        best_error = np.inf
        iter = 10000
        sigma = 0.1 ## 阈值
        pretotal = 0 ## 内点个数
        Per = 0.99 ## 正确概率
        plane = Plane(np.array([0, 0, 1]), 0)
        for _ in range(iter):
            A, B, C = data[np.random.choice(len(data), 3, replace=False)]
            plane.solve_plane(A, B, C)
            total_inlier = 0
            error = 0
            for point in data:
                if plane.solve_distance(point) < sigma:
                    total_inlier += 1
                error += plane.solve_distance(point) ** 2
            if total_inlier > pretotal:
                iters = np.log(1 - Per) / np.log(1 - pow(total_inlier / len(data), 3))
                pretotal = total_inlier
            if error < best_error:
                best_error = error
                best_plane = plane
        return best_plane
    
def ToF_RANSAC(data, expansion_res=256):
    """
    ToF数据平面拟合
    :params: data: 3D points
    :return: best_plane: 最优平面
    """
    expansion_times = expansion_res // 8
    expansion_data = np.zeros((expansion_res, expansion_res))
    for i in range(8):
        for j in range(8):
            expansion_data[i*expansion_times:(i+1)*expansion_times, j*expansion_times:(j+1)*expansion_times] = data[i][j]
    
    plane = Plane(np.array([0, 0, 1]), 0)
    best_plane = plane.RANSAC(data)
    return best_plane