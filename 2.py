import sys
import numpy as np
import cv2 as cv
import math

def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param[0].append([x, y])

def mouse_click(img):
    points = []
    WINDOW_NAME = 'Please click the corners.'
    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, on_mouse, [points])

    img_ = img.copy()
    while True:
        for i, p in enumerate(points):
            # draw points on img_
            cv.circle(img_, tuple(p), 4, (0, 0, 255), -1)
        cv.imshow(WINDOW_NAME, img_)
        cv.waitKey(20)
        if len(points) == 4:
            cv.circle(img_, tuple(points[3]), 4, (0, 0, 255), -1)
            break
    cv.destroyAllWindows()

    cv.imshow('The Corners', img_)
    cv.waitKey(8000)
    cv.destroyAllWindows()
    return points

def normalize(points):
    mean, std = np.mean(points, 0), np.std(points)
    T = np.array([[std/np.sqrt(2), 0, mean[0]],
                  [0, std/np.sqrt(2), mean[1]],
                  [0, 0, 1]])
    T = np.linalg.inv(T)

    points = T.dot(np.concatenate((points.T, np.ones((1, points.shape[0])))))
    points = points[0:2].T
    return points, T

def DLT(points1, points2):
    A = []
    for i in range(points1.shape[0]):
        u , v, u_p, v_p = points1[i][0], points1[i][1], points2[i][0], points2[i][1]
        A.append([u, v, 1, 0, 0, 0, -u_p*u, -u_p*v, -u_p])
        A.append([0, 0, 0, u, v, 1, -v_p*u, -v_p*v, -v_p])
    U, S, VH = np.linalg.svd(A)
    H = VH[-1, :].reshape([3, 3])
    return H

def backward_warping(img, H, w, h):
    warp_img = np.zeros((h, w, 3), np.uint8)
    for i in range(h):
        for j in range(w):
            x, y, z = H.dot(np.array([i, j, 1]))
            x /= z
            y /= z
            warp_img[i, j] = bilinear_interpolation(img, x, y)

    return warp_img

def bilinear_interpolation(img, x, y):
    x_floor = math.floor(x)
    x_ceil = math.ceil(x)
    y_floor = math.floor(y)
    y_ceil = math.ceil(y)

    a, b, c, d = img[y_floor, x_floor], img[y_floor, x_ceil], img[y_ceil, x_ceil], img[y_ceil, x_floor]
    wa = (x_ceil - x) * (y_ceil - y)
    wb = (x - x_floor) * (y_ceil - y)
    wc = (x - x_floor) * (y - y_floor)
    wd = (x_ceil - x) * (y - y_floor)
    return wa * a + wb * b + wc * c + wd * d

if __name__ == '__main__':
    img = cv.imread(sys.argv[1])
    img = cv.resize(img, (0, 0), fx = 0.3, fy = 0.3)
    height, width = img.shape[0], img.shape[1]
    points1 = np.array(mouse_click(img))
    points2 = np.array([[0, 0], [0, width-1], [height-1, 0], [height-1, width-1]])

    points1_norm, T1 = normalize(points1)
    points2_norm, T2 = normalize(points2)

    H_hat = DLT(points1_norm, points2_norm)
    H_norm = np.linalg.inv(T2).dot(H_hat).dot(T1)
    H_norm_inv = np.linalg.inv(H_norm)

    warp_img = backward_warping(img, H_norm_inv, width, height)

    cv.imshow('warp', warp_img)
    cv.waitKey(0)