import sys
import numpy as np
import cv2 as cv
import random

def get_correspondences(match_method, img1, img2):

    if match_method == 'sift':
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        matcher = cv.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        good_matches = sorted(good_matches, key=lambda x: x.distance)
        good_matches = good_matches[:50]

        points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
        points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    elif match_method == 'orb':
        orb = cv.ORB_create()
        img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        kp1, des1 = orb.detectAndCompute(img1_gray, None)
        kp2, des2 = orb.detectAndCompute(img2_gray, None)

        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
        matches = matcher.match(des1, des2)

        good_matches = sorted(matches, key = lambda x : x.distance)
        good_matches = good_matches[:50]

        points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
        points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    return points1, points2, kp1, kp2, good_matches

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

def error_computation(p_t, p_t_hat):
    diff = (p_t - p_t_hat)**2
    diff = diff.sum(axis = 1)
    error_vector = np.sqrt(diff)
    error = error_vector.sum()/p_t.shape[0]

    return error_vector, error

def ransac(points1, points2, p_s, p_t, threshold, k, N):
    num_best_inliners = 0

    for i in range(N):
        chosen_idx = random.sample(range(len(points1)), k)
        chosen_points1 = np.array([points1[i] for i in chosen_idx])
        chosen_points2 = np.array([points2[i] for i in chosen_idx])
        
        H = DLT(chosen_points1, chosen_points2)

        p_t_hat = H.dot(np.concatenate((p_s.T, np.ones((1, p_s.shape[0]))))) 
        p_t_hat = (p_t_hat / p_t_hat[2, :])[:2].T
        error_vector, error = error_computation(p_t, p_t_hat)

        # Normalized points

        points1_norm, T1 = normalize(chosen_points1)
        points2_norm, T2 = normalize(chosen_points2)

        H_hat = DLT(points1_norm, points2_norm)
        H_norm = np.linalg.inv(T2).dot(H_hat).dot(T1)

        p_t_hat = H_norm.dot(np.concatenate((p_s.T, np.ones((1, p_s.shape[0])))))
        p_t_hat = (p_t_hat / p_t_hat[2, :])[:2].T
        error_norm_vector, error_norm = error_computation(p_t, p_t_hat)
        
        idx = np.where(error_vector < threshold)[0]
        num_inliers = len(idx)
        
        if num_inliers > num_best_inliners:
            num_best_inliers = num_inliers
            best_idx = chosen_idx
            best_H = H.copy()
            best_H_norm = H_norm.copy()
            lowest_error = error
            lowest_error_norm = error_norm
    return best_idx, best_H, best_H_norm, lowest_error, lowest_error_norm

if __name__ == '__main__':
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])

    gt_correspondences = np.load(sys.argv[3])
    p_s = gt_correspondences[0]
    p_t = gt_correspondences[1]

    match_methods = ['sift', 'orb']
    threshold = 1.2
    k = int(input('Please input value k: '))
    N = 100000

    for match_method in match_methods:
        points1, points2, kp1, kp2, good_matches = get_correspondences(match_method, img1, img2)
        idx, H, H_norm, error, error_norm = ransac(points1, points2, p_s, p_t, threshold, k, N)
        chosen_points1 = np.array([points1[i] for i in idx])
        chosen_points2 = np.array([points2[i] for i in idx])

        print('Match method:', match_method)
        print('Chosen points in image1:')
        print(chosen_points1)
        print('Chosen points in image2:')
        print(chosen_points2)

        print('Matrix H:')
        print(H)
        print('Matrix H with points normalization:')
        print(H_norm)

        print('DLT error:', error)
        print('Normalized DLT error:', error_norm, '\n')

        matches = []
        for i in idx:
            matches.append(good_matches[i])
        img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img_draw_match = cv.resize(img_draw_match, (0, 0), fx = 0.5, fy = 0.5)
        if match_method == 'sift':
            cv.imshow('SIFT match', img_draw_match)
        else:
            cv.imshow('ORB match', img_draw_match)
            cv.waitKey(0)