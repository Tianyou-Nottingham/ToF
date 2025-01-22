import cv2
import numpy as np
import pyrealsense2 as rs
import os
import time
import serial
from TOF_RANSAC import Plane
from read_data_utils import read_serial_data, refine_by_time, visualize2D
import configs.config as cfg
import matplotlib.pyplot as plt
from obstacle_avoidance import outliers_detection
from utils.distance_rectified_fov import distance_rectified_fov
from two_plane_fit import two_plane_visualization
import re

RGB_path = r"E:\Projects\ToF\ToF\output\RGB1_Color.png"
Depth_path = r"C:\Users\ezxtz6\Pictures\Depth0.png"
Intrinsic = cfg.RealSense["K"]
chessboard_size = [7, 10]


def normalize(value, vmin=0.0, vmax=4.0):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.0
    value = (value * 255.0).astype(np.uint8)
    return value


def corner_detection(img):
    Grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objpoints = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objpoints[:, :2] = np.mgrid[
        0 : chessboard_size[0], 0 : chessboard_size[1]
    ].T.reshape(-1, 2)
    points_w = []
    points_i = []
    ret, corners = cv2.findChessboardCorners(Grey, chessboard_size, None)
    print(f"Corner Detection: {ret}")
    if ret:
        cv2.cornerSubPix(
            Grey,
            corners,
            chessboard_size,
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        points_w.append(objpoints)
        points_i.append(corners)
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        # cv2.imshow("RGB", img)
        # cv2.imwrite("./RGB_corners.png", img)
        # cv2.waitKey(0)
    return ret, img, points_w, points_i


def find_line(img, corners, chessboard_size):
    w, l = chessboard_size
    lines_w = []
    lines_l = []
    for i in range(l):
        line = [corners[i * w], corners[(i + 1) * w - 1]]
        lines_w.append(line)  # 短边
    for j in range(w):
        line = [corners[j], corners[(l - 1) * w + j]]
        lines_l.append(line)  # 长边
    # Draw the lines
    for line in lines_w:
        img = cv2.line(
            img, line[0].astype(np.uint), line[-1].astype(np.uint), (0, 255, 0), 2
        )
    for line in lines_l:
        img = cv2.line(
            img, line[0].astype(np.uint), line[-1].astype(np.uint), (0, 0, 255), 2
        )
    return img, lines_w, lines_l


def find_infinity_point(img, lines):
    """
    计算两条平行线的无穷远点
    :param img:
    :param lines: 平行线的两个端点
    :return: 无穷远点的齐次坐标 (x, y, 0)

    ## In camera projection space, the intersection is the same as the vanishing point
    ## We can use the cross product to find the intersection
    ## The intersection of two lines is the vanishing point
    """
    w, h = chessboard_size
    lines_projection = []
    for line in lines:
        A = np.array([line[0][0], line[0][1], 1])
        B = np.array([line[-1][0], line[-1][1], 1])
        lines_projection.append(np.cross(A, B))
    ## RANSAC to find the intersection
    ## The intersection is the vanishing point
    best_point = [0, 0, 0]
    best_error = np.inf
    max_iter = 1000
    pretotal = 0
    k = 0
    total_inlier = 0
    while (k < max_iter) and (total_inlier < len(lines_projection) * 2 / 3):
        [A_index, B_index] = np.random.choice(len(lines_projection), 2, replace=False)
        A_line = lines_projection[A_index]
        B_line = lines_projection[B_index]
        point = np.cross(A_line, B_line)
        error = 0
        total_inlier = 0
        for i in range(len(lines_projection)):
            if np.linalg.norm(np.cross(point, lines_projection[i])) < 1e-3:
                total_inlier += 1
            error += np.linalg.norm(np.cross(point, lines_projection[i]))
        if error < best_error:
            best_error = error
            best_point = point
        k += 1
    return best_point / best_point[-1]


def rs_capture_align(save=True):
    # 创建realsense pipeline 以及 serial
    pipeline = rs.pipeline()

    ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
    last_distances = np.zeros((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    last_sigma = np.ones((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    # Create a config并配置要流​​式传输的管道
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)
    # 将depth对齐到color
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 保存的图片和实时的图片界面
    cv2.namedWindow("RealSense live", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("ToF live", cv2.WINDOW_AUTOSIZE)

    if save == True:
        # 按照日期创建文件夹
        save_path = os.path.join(
            os.getcwd(), "calib", time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        )
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            os.mkdir(os.path.join(save_path, "color"))
            os.mkdir(os.path.join(save_path, "depth"))
        cv2.namedWindow("save", cv2.WINDOW_AUTOSIZE)
    saved_color_image = None  # 保存的临时图片
    saved_depth_mapped_image = None
    saved_count = 0

    # 主循环
    try:
        while True:
            #### 1.1 Read RealSense data ####
            frames = pipeline.wait_for_frames()
            profile = pipeline.get_active_profile()

            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_data = np.asanyarray(aligned_depth_frame.get_data(), dtype="float16")
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            scaled_depth_image = depth_image * depth_scale
            color_image = np.asanyarray(color_frame.get_data())
            depth_mapped_image = cv2.applyColorMap(
                normalize(scaled_depth_image),
                cv2.COLORMAP_MAGMA,
            )

            #### 1.2 Read ToF data ####
            distances, sigma = read_serial_data(ser, cfg.Sensor["resolution"])
            time_refine_distances, time_refine_sigma = refine_by_time(
                distances, sigma, last_distances, last_sigma
            )
            last_distances = distances
            last_sigma = sigma

            ToF_depth_map, ToF_sigma = visualize2D(
                time_refine_distances,
                sigma,
                cfg.Sensor["resolution"],
                cfg.Sensor["output_shape"],
            )
            ToF_depth_map = cv2.applyColorMap(
                normalize(ToF_depth_map), cv2.COLORMAP_MAGMA
            )

            ##### 2. Image and ToF data processing #####
            #### 2.1 ToF data processing ####
            grad_x = np.gradient(time_refine_distances, axis=1)
            grad_y = np.gradient(time_refine_distances, axis=0)
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
            plane1_index = outliers_detection(plane1_index, 4)
            plane2_index = outliers_detection(plane2_index, 4)
            points_plane1 = np.array(
                [
                    [i, j, time_refine_distances[i, j]]
                    for i, j in plane1_index
                    if time_refine_distances[i, j] > 1.5
                    and time_refine_distances[i, j] < 1000
                ]
            )
            points_plane2 = np.array(
                [
                    [i, j, time_refine_distances[i, j]]
                    for i, j in plane2_index
                    if time_refine_distances[i, j] > 1.5
                    and time_refine_distances[i, j] < 1000
                ]
            )

            #### 2.2 Realsense processing ####
            color_blur = cv2.GaussianBlur(color_image, (0, 0), 5)
            color_usm = cv2.addWeighted(color_image, 1.5, color_blur, -0.5, 0)
            corner_detection_ret, corner_detection_img, points_w, points_i = (
                corner_detection(color_usm)
            )
            points_w, points_i = np.array(points_w), np.array(points_i)
            points_w = points_w.reshape((-1, 3))
            points_i = points_i.reshape((-1, 2))
            print(f"Shape of w is {points_w.shape}, shape of i is {points_i.shape}")
            if corner_detection_ret == True:
                ##### 3 Prepare ToF plane fitting #####
                plane1 = Plane(np.array([0, 0, 1]), 0)
                plane2 = Plane(np.array([0, 0, 1]), 0)
                plane3 = Plane(np.array([0, 0, 1]), 0)
                ##### 4. Visualization #####
                cv2.imshow(
                    "RealSense live",
                    np.hstack((corner_detection_img, depth_mapped_image)),
                )

                cv2.imshow("ToF live", ToF_depth_map)
                key = cv2.waitKey(10)

                #### 3.1 Plane fitting ####
                plane1 = plane1.ToF_RANSAC(points_plane1, res=cfg.Sensor["resolution"])
                plane2 = plane2.ToF_RANSAC(points_plane2, res=cfg.Sensor["resolution"])

                fig = plt.figure(figsize=(14, 7))

                plane1.ToF_visualization(
                    fig,
                    time_refine_distances,
                    time_refine_sigma,
                    cfg.Sensor["resolution"],
                    cfg.Sensor["output_shape"],
                )

                if cfg.Code["distance_rectified_fov"]:
                    points1 = distance_rectified_fov(points_plane1)
                    points2 = distance_rectified_fov(points_plane2)

                two_plane_visualization(fig, plane1, plane2, points1, points2)

                ### 3.2 Realsense vanishing point calculation ####
                ret, rvec, tvec = cv2.solvePnP(
                    points_w.astype("float32"),
                    points_i.astype("float32"),
                    Intrinsic,
                    None,
                )
                R, _ = cv2.Rodrigues(rvec)
                points_c = R @ points_w.T + tvec
                points_c = points_c.T
                plane3 = plane3.fit_plane(points_c)
                print(f"Plane3: N: {plane3.N}, d:{plane3.d}, error:{plane3.error}")
                line_image, lines_w, lines_l = find_line(
                    color_usm, points_i, chessboard_size
                )
                vanishing_point_w = find_infinity_point(color_usm, lines_w)
                vanishing_point_l = find_infinity_point(color_usm, lines_l)
                # print(
                #     f"Vanishing Point W: {vanishing_point_w}, L: {vanishing_point_l}\n"
                # )
                print(
                    f"Plane1: N: {plane1.N}, d:{plane1.d}, error:{plane1.error}, Plane2: N: {plane2.N}, d:{plane2.d},error:{plane2.error}\n"
                )

                # Save the plane fitting result and vanishing points to .txt file
                def on_key_press(event):
                    if event.key == "p":
                        txt_name = os.path.join(save_path, "plane_fitting.txt")
                        with open(txt_name, "a") as f:
                            f.write(
                                f"Vanishing Point W: {vanishing_point_w}, L: {vanishing_point_l}. Plane1: N: {plane1.N}, d:{plane1.d}, error:{plane1.error}; Plane2: N: {plane2.N}, d:{plane2.d},error:{plane2.error}; plane3c: N: {plane3.N}, d:{plane3.d},error:{plane3.error}\n"
                            )
                        pass
                    else:
                        pass

                # 将回调函数与图形对象绑定
                fig.canvas.mpl_connect("key_press_event", on_key_press)
                plt.show()

            ##### 4. Visualization #####
            else:
                cv2.imshow(
                    "RealSense live", np.hstack((color_image, depth_mapped_image))
                )
                cv2.imshow("ToF live", ToF_depth_map)
                key = cv2.waitKey(30)

            if save == True:
                # s 保存图片
                if key & 0xFF == ord("s"):
                    saved_color_image = color_image
                    saved_depth_mapped_image = depth_mapped_image

                    # 彩色图片保存为png格式
                    cv2.imwrite(
                        os.path.join(
                            (save_path), "color", "{}.png".format(saved_count)
                        ),
                        saved_color_image,
                    )
                    # 深度信息由采集到的float16直接保存为npy格式
                    np.save(
                        os.path.join((save_path), "depth", "{}".format(saved_count)),
                        depth_data,
                    )
                    saved_count += 1
                    # cv2.imshow(
                    #     "save", np.hstack((saved_color_image, saved_depth_mapped_image))
                    # )

            # q 退出
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


def main():
    Rgb = cv2.imread(RGB_path)
    # 上下左右翻转图片
    Rgb = cv2.flip(Rgb, -1)
    Rgb_blur = cv2.GaussianBlur(Rgb, (0, 0), 5)
    usm = cv2.addWeighted(Rgb, 1.5, Rgb_blur, -0.5, 0)
    cv2.imshow("RGB", usm)
    cv2.waitKey(0)
    print(Rgb.shape)
    ## 1. Corner Detection
    points_w, points_i = corner_detection(usm)

    ## 2. Find the lines
    line_img, lines_horizontal, lines_vertical = find_line(
        usm, points_i[0], chessboard_size
    )
    cv2.imshow("RGB", line_img)
    cv2.imwrite("./RGB_lines.png", line_img)
    cv2.waitKey(0)

    ## 3. Find the intersection
    # change the points into projection space
    # points_i = np.array(points_i).reshape((chessboard_size[1], chessboard_size[0], 2))
    vanishing_point_horizontal = find_infinity_point(Rgb, lines_horizontal)
    vanishing_point_vertical = find_infinity_point(Rgb, lines_vertical)
    if (
        vanishing_point_horizontal[0] < Rgb.shape[0]
        and vanishing_point_horizontal[1] < Rgb.shape[1]
    ):
        cv2.circle(
            Rgb,
            (int(vanishing_point_horizontal[0]), int(vanishing_point_horizontal[1])),
            5,
            (0, 255, 0),  ## BGR
            -1,
        )
    if (
        vanishing_point_vertical[0] < Rgb.shape[0]
        and vanishing_point_vertical[1] < Rgb.shape[1]
    ):
        cv2.circle(
            Rgb,
            (int(vanishing_point_vertical[0]), int(vanishing_point_vertical[1])),
            5,
            (0, 0, 255),
            -1,
        )
    cv2.imshow("RGB", Rgb)
    cv2.imwrite("./RGB_vanishing_point.png", Rgb)
    cv2.waitKey(0)
    print(f"Vanishing Point Horizontal: {vanishing_point_horizontal}")
    print(f"Vanishing Point Vertical: {vanishing_point_vertical}")


def read_N_and_Vp(file_path):
    # <built-in function localtime>:
    # Vanishing Point Horizontal: [-561.14966641  510.21300619    1.        ],
    # Vertical: [659.72555736 402.49584642   1.        ].
    # Plane1: N: [-0.18446338 -0.36972649  0.91064569], d:446.8080933309752, error:0.002568954238524618,
    # Plane2: N: [-0.07672262  0.78625689  0.61311805], d:369.9294947441689,error:7.191378483069497e-05
    vp_w = []
    vp_l = []
    plane1_N = []
    plane2_N = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = re.split("\[|\]", line)
            vp_w.append([float(i) for i in temp[1].split()])
            vp_l.append([float(i) for i in temp[3].split()])
            plane1_N.append([float(i) for i in temp[5].split()])
            plane2_N.append([float(i) for i in temp[7].split()])
    vp_w = np.array(vp_w)
    vp_l = np.array(vp_l)
    plane1_N = np.array(plane1_N)
    plane2_N = np.array(plane2_N)
    # print(f"Vanishing Point Horizontal: {vp_horizontal}")
    # print(f"Vanishing Point Vertical: {vp_vertical}")
    # print(f"Plane1: N: {plane1_N}")
    # print(f"Plane2: N: {plane2_N}")
    return vp_w, vp_l, plane1_N, plane2_N


def calib_by_N_and_Vp(cfg, Vp1, Vp2, N1, N2):
    """
    通过N和Vp计算相机内参
    """
    # 交换N的x y
    N1 = N1[:, [1, 0, 2]]
    N2 = N2[:, [1, 0, 2]]
    K = cfg.RealSense["K"]
    K_inv = np.linalg.inv(K)
    K_Vp1 = K_inv @ Vp1.T
    K_Vp2 = K_inv @ Vp2.T
    d1_orient_vec = K_Vp1 / np.linalg.norm(K_Vp1, axis=0)
    d2_orient_vec = K_Vp2 / np.linalg.norm(K_Vp2, axis=0)
    # 由于RealSense 是倒着放的，所以Camera坐标系差一个绕z轴的180度
    # 实际操作的时候，RealSense的color image和depth都应该反转过来。
    # 可以相当于将Realsense 的坐标系乘上一个[[-1, 0, 0],[0, 1, 0], [0, 0, -1]]
    # 这里没有翻转图像，所以R乘上一个[[-1, 0, 0],[0, 1, 0], [0, 0, -1]]
    R_reverse = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    d = np.hstack((d1_orient_vec, d2_orient_vec))
    print(f"d shape: {d.shape}")
    N = np.vstack((N2, N1))
    print(f"N shape: {N.shape}")
    S = N.T @ d.T
    print(f"S shape: {S.shape}")
    U, S_, V = np.linalg.svd(S)
    R = V @ U.T @ R_reverse
    # 矫正R的行列式为1
    # if np.linalg.det(R) < 0:
    #     R[:, -1] = -R[:, -1]
    print(f"R: {R}")
    return R


if __name__ == "__main__":
    rs_capture_align()
    # file_path = r"D:\Downloads\ToF\calib\2025_01_20_21_04_24\plane_fitting.txt"
    # Vp1, Vp2, N1, N2 = read_N_and_Vp(file_path)
    # R = calib_by_N_and_Vp(cfg, Vp1, Vp2, N1, N2)
    # Visualization of the rotation matrix
