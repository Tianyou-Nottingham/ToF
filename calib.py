import cv2
import numpy as np
import pyrealsense2 as rs
import os
import time
import serial
from TOF_RANSAC import Plane
from read_data_utils import read_serial_data, refine_by_time, visualize2D, normalize
import configs.config as cfg
import matplotlib.pyplot as plt
from obstacle_avoidance import outliers_detection
from utils.distance_rectified_fov import distance_rectified_fov
from two_plane_fit import two_plane_visualization

RGB_path = r"E:\Projects\ToF\ToF\output\RGB1_Color.png"
Depth_path = r"C:\Users\ezxtz6\Pictures\Depth0.png"
Intrinsic = np.array([[617.173, 0, 320.5], [0, 617.173, 240.5], [0, 0, 1]])
chessboard_size = [7, 10]


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
    w, h = chessboard_size
    lines_horizontal = []
    lines_vertical = []
    for i in range(h):
        line = [corners[i * w][0], corners[(i + 1) * w - 1][0]]
        lines_horizontal.append(line)
    for j in range(w):
        line = [corners[j][0], corners[(h - 1) * w + j][0]]
        lines_vertical.append(line)
    # Draw the lines
    for line in lines_horizontal:
        img = cv2.line(
            img, line[0].astype(np.uint), line[-1].astype(np.uint), (0, 255, 0), 2
        )
    for line in lines_vertical:
        img = cv2.line(
            img, line[0].astype(np.uint), line[-1].astype(np.uint), (0, 0, 255), 2
        )
    return img, lines_horizontal, lines_vertical


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

    align_to = rs.stream.color
    align = rs.align(align_to)

    if save == True:
        # 按照日期创建文件夹
        save_path = os.path.join(
            os.getcwd(), "calib", time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        )
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            os.mkdir(os.path.join(save_path, "color"))
            os.mkdir(os.path.join(save_path, "depth"))

    # 保存的图片和实时的图片界面
    cv2.namedWindow("RealSense live", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("ToF live", cv2.WINDOW_AUTOSIZE)
    if save == True:
        cv2.namedWindow("save", cv2.WINDOW_AUTOSIZE)
    saved_color_image = None  # 保存的临时图片
    saved_depth_mapped_image = None
    saved_count = 0

    # 主循环
    try:
        while True:
            ## 1. Read the data
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_data = np.asanyarray(aligned_depth_frame.get_data(), dtype="float16")
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_mapped_image = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=1),
                # depth_image,
                cv2.COLORMAP_MAGMA,
            )

            # 1.2 Read ToF data
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
            ToF_depth_map = cv2.applyColorMap(ToF_depth_map, cv2.COLORMAP_MAGMA)
            # 2. Image and ToF data processing
            # 2.1 ToF data processing
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

            # 2.2 Realsense processing
            color_blur = cv2.GaussianBlur(color_image, (0, 0), 5)
            color_usm = cv2.addWeighted(color_image, 1.5, color_blur, -0.5, 0)
            corner_detection_ret, corner_detection_img, points_w, points_i = (
                corner_detection(color_usm)
            )

            if corner_detection_ret == True:
                # 3 Prepare ToF plane fitting
                plane1 = Plane(np.array([0, 0, 1]), 0)
                plane2 = Plane(np.array([0, 0, 1]), 0)
                # 4. Visualization
                cv2.imshow(
                    "RealSense live",
                    np.hstack((corner_detection_img, depth_mapped_image)),
                )

                cv2.imshow("ToF live", ToF_depth_map)
                key = cv2.waitKey(10)

                # 3.1 Plane fitting
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

                def on_key_press(event):
                    if event.key == "p":
                        txt_name = os.path.join(save_path, "plane_fitting.txt")
                        with open(txt_name, "a") as f:
                            f.write(
                                f"{time.localtime}: Vanishing Point Horizontal: {vanishing_point_horizontal}, Vertical: {vanishing_point_vertical}. Plane1: N: {plane1.N}, d:{plane1.d}, error:{plane1.error}, Plane2: N: {plane2.N}, d:{plane2.d},error:{plane2.error}\n"
                            )
                        pass
                    else:
                        pass

                # 将回调函数与图形对象绑定
                fig.canvas.mpl_connect("key_press_event", on_key_press)
                plt.show()

                # 3.2 Realsense vanishing point calculation
                line_image, lines_horizontal, lines_vertical = find_line(
                    color_usm, points_i[0], chessboard_size
                )
                vanishing_point_horizontal = find_infinity_point(
                    color_usm, lines_horizontal
                )
                vanishing_point_vertical = find_infinity_point(
                    color_usm, lines_vertical
                )
                print(
                    f"Vanishing Point Horizontal: {vanishing_point_horizontal}, Vertical: {vanishing_point_vertical}\n"
                )
                print(
                    f"Plane1: N: {plane1.N}, d:{plane1.d}, error:{plane1.error}, Plane2: N: {plane2.N}, d:{plane2.d},error:{plane2.error}\n"
                )

            # 4. Visualization
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


if __name__ == "__main__":
    rs_capture_align()
