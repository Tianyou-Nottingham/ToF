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
from read_data_utils import read_serial_data
import numpy as np
import cv2
import time
import serial
import configs.config as cfg
from direction_visualization import refine_by_time
from read_data_utils import visualize2D, normalize
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Plane:
    def __init__(self, N, d) -> None:
        self.N = N
        self.d = d
        self.error = 0

    def solve_distance(self, point):
        """
        平面方程: N * x + d = 0
        """
        distance = np.linalg.norm(np.dot(self.N, point) - self.d)
        return distance
    
    def solve_plane(self, A, B, C):
        """
        求解平面方程
        :params: three points
        :return: Nx(平面法向量), d
        """
        Nx = np.cross(B - A, C - A)
        Nx = Nx / np.linalg.norm(Nx)
        d = -np.dot(Nx, A+B+C) / 3
        self.N = Nx if Nx[2] > 0 else -Nx
        self.d = d if Nx[2] > 0 else -d

    def fit_plane(self, pts, initial_est=[0, 0, 1, 0.5]):
        """
        Fit a plane given by ax+d = 0 to a set of points
        Works by minimizing the sum over all points x of ax+d
        Ars:
            pts: array of points in 3D space
        Returns:
            (3x1 numpy array): a vector for plane equation
            (float): d in plane equation
            (float): sum of residuals for points to plane (orthogonal l2 distance)
        """

        pts = np.array(pts)

        def loss_fn(x, points):
            self.N = np.array(x[:3])
            self.d = x[3]

            loss = 0
            for point in points:
                loss += np.abs(np.dot(self.N, np.array(point)) - self.d)

            return loss

        def a_constraint(x):
            return np.linalg.norm(x[:3]) - 1

        soln = minimize(
            loss_fn,
            np.array(initial_est),
            args=(pts),
            method="slsqp",
            constraints=[{"type": "eq", "fun": a_constraint}],
            bounds=[(-1, 1), (-1, 1), (-1, 1), (0, None)],
        )

        self.N = soln.x[:3]
        self.d = soln.x[3]
        self.error = soln.fun

        return self.N, self.d, self.error

    def plane_visualization(self,fig, N, d, data, color='r'):
        """
        平面可视化
        :param N: 平面的法向量(3维,形如 [a, b, c])
        :param d: 平面基点偏移量(3维向量,形如 [d_x, d_y, d_z])
        平面方程: N · X + d = 0 或 ax + by + cz + d_offset = 0
        """
        # 解析法向量
        a, b, c = N

        # 创建网格 (x, y)
        x = np.linspace(0, 8, 100)
        y = np.linspace(0, 8, 100)
        X, Y = np.meshgrid(x, y)

        # 根据平面方程 N · X + d = 0 求解 z
        if c == 0:
            raise ValueError("The normal vector's z component (c) cannot be zero for visualization.")
        Z =  -(a * X + b * Y - d) / c

        # 创建 3D 图形
        ax = fig.add_subplot(121, projection='3d')

        # 绘制平面
        # if Z < 0:
        #     ax.plot_surface(-X, -Y, -Z, alpha=0.5, color='blue', edgecolor='k')
        # else:
        ax.plot_surface(X, Y, Z, alpha=0.5, color='blue', edgecolor='k')

        # 绘制数据点
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, marker='o')
        ax.view_init(elev=50, azim=0)
        # 设置轴标签
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        # 设置标题
        ax.set_title('Plane Visualization')

        # 显示图形
        # plt.show()

    def ToF_visualization(self, fig, distance, sigma, res=8, expansion_res=256):
        depth, sigma = visualize2D(distance, sigma, res, expansion_res)
        ax = fig.add_subplot(122)
        norm = plt.Normalize(0, 2500)
        ax.imshow(depth, cmap='magma')
        ax.set_title('Depth Image')
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='magma'), ax=ax)
        # ax = fig.add_subplot(122)
        # ax.imshow(color_sigma)
        # ax.set_title('Sigma Image')
        # plt.show()

    def ToF_RANSAC(self, data, res=8, expansion_res=32):
        """
        ToF数据平面拟合
        :params: data: obstacl or safe 3D points
        :params: res: resolution of the ToF data
        :params: expansion_res: resolution of the expansion data
        :return: best_plane: 最优平面
        """
        if len(data) < 3:
            raise ValueError("The number of points should be more than 3.")
        
        pad_size = expansion_res // res
        # expansion_data = np.zeros((pad_size, pad_size))
        [x_index, y_index, d_value] = [data[:, 0], data[:, 1], data[:, 2]]
        # for i in range(len(data)):
        #     expansion_data[int(x_index[i]*pad_size): int((x_index[i]+1)*pad_size), 
        #                    int(y_index[i]*pad_size): int((y_index[i]+1)*pad_size)] = d_value[i]

        best_plane = None
        best_error = np.inf
        max_error = 1000
        max_iter = 1000
        sigma = 100 ## 阈值
        pretotal = 0 ## 内点个数
        Per = 50000 ## 正确概率
        k = 0
        while k < max_iter and best_error > max_error: #pretotal < len(data) *2/3: ## 当内点个数大于总点数的2/3 或 大于预设迭代次数时，停止迭代
            point_offset = [np.random.choice(pad_size, len(data)), np.random.choice(pad_size, len(data))]
            points = []
            for i in range(len(data)):
                point = np.array([x_index[i]*pad_size+point_offset[0][i],
                                  y_index[i]*pad_size+point_offset[1][i], d_value[i]])
                points.append(point)

            self.fit_plane(points)
            # [A_index, B_index, C_index] = data[np.random.choice(len(data), 3, replace=False)]
            # A = np.array([A_index[0]*pad_size + np.random.choice(pad_size, 1)[0], 
            #               A_index[1]*pad_size + np.random.choice(pad_size, 1)[0], A_index[2]])
            # B = np.array([B_index[0]*pad_size + np.random.choice(pad_size, 1)[0], 
            #               B_index[1]*pad_size + np.random.choice(pad_size, 1)[0], B_index[2]])
            # C = np.array([C_index[0]*pad_size + np.random.choice(pad_size, 1)[0], 
            #               C_index[1]*pad_size + np.random.choice(pad_size, 1)[0], C_index[2]])
            # # self.solve_plane(A, B, C)
            # self.fit_plane([A, B, C])
            # total_inlier = 0
            # error = 0
            # ## 只取一个点进行RANSAC
            # point_offset = [np.random.choice(pad_size, len(data)), np.random.choice(pad_size, len(data))]
            # for i in range(len(data)):
            #     point = np.array([x_index[i]*pad_size+point_offset[0][i],
            #                        y_index[i]*pad_size+point_offset[1][i], d_value[i]])
            #     if self.solve_distance(point) < sigma:
            #         total_inlier += 1
            #     error += self.solve_distance(point) ** 2

            # if total_inlier > pretotal:
            #     # iters = np.log(1 - Per) / np.log(1 - pow(total_inlier / len(data), 3))
            #     pretotal = total_inlier
            if self.error < best_error:
                best_error = self.error
                best_plane = self
            k += 1
            # # print(f"iter: {k}, total_inlier: {total_inlier}, best_error: {best_error}")
        return best_plane, self.error #np.sqrt(best_error / total_inlier)

def test():
    ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
    last_distances = np.zeros((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    last_sigma = np.ones((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    points3D = []
    while True:
        ## 1. Read the data from the serial port
        distances, sigma = read_serial_data(ser, cfg.Sensor["resolution"])
        
        ## 2. Refine by time
        time_refine_distances, time_refine_sigma = refine_by_time(distances, sigma, last_distances, last_sigma)
        # print(time_refine_distances)
        points3D = np.array([[i, j, time_refine_distances[i, j]] for i in range(cfg.Sensor["resolution"]) for j in range(cfg.Sensor["resolution"])])
        last_distances = distances
        last_sigma = sigma

        # ## 3. ToF RANSAC
        # plane = Plane(np.array([0, 0, 1]), 0)
        # plane.ToF_RANSAC(points3D, cfg.Sensor["resolution"], 256)
        plane = Plane(np.array([0, 0, 1]), 0)
        plane.fit_plane(points3D)
        ## 4. Visualization

        depth, sigma = visualize2D(time_refine_distances, sigma, cfg.Sensor["resolution"], cfg.Sensor["output_shape"])
        print(f"Plane N: {plane.N}, d: {plane.d}. Error: {plane.error}")
        color_depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA) 
        fig = plt.figure(figsize=(14, 7))
        plane.plane_visualization(fig, plane.N, plane.d, points3D)
        plane.ToF_visualization(fig, time_refine_distances, sigma, cfg.Sensor["resolution"], cfg.Sensor["output_shape"])
        plt.show()
        plt.close(fig)
        # cv2.imshow('depth', color_depth)
        # cv2.waitKey(1) & 0xFF == ord('q')  


if __name__ == "__main__":
    test()