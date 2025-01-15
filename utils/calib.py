import cv2
import numpy as np

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
        cv2.imshow("RGB", img)
        cv2.imwrite("./RGB_corners.png", img)
        cv2.waitKey(0)
    return points_w, points_i


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


if __name__ == "__main__":
    Rgb = cv2.imread(RGB_path)
    # 上下左右翻转图片
    Rgb = cv2.flip(Rgb, -1)
    cv2.imshow("RGB", Rgb)
    cv2.waitKey(0)
    print(Rgb.shape)
    ## 1. Corner Detection
    points_w, points_i = corner_detection(Rgb)

    ## 2. Find the lines
    line_img, lines_horizontal, lines_vertical = find_line(
        Rgb, points_i[0], chessboard_size
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
