import open3d as o3d
import numpy as np
from numpy.linalg import norm
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp

class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    #TODO:
                    # insert new camera pose here using vis.add_geometry()
                    pass
            except: pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def process_frames(self, queue):
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        orb = cv.ORB_create()
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        img1 = cv.imread(self.frame_paths[0])
        img1 = cv.undistort(img1, self.K, self.dist)
        img1_g = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', img1)

        for frame_path in self.frame_paths[1:]:
            img2 = cv.imread(frame_path)
            img2 = cv.undistort(img2, self.K, self.dist)
            img2_g = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
            #TODO: compute camera pose here
            kp1, des1 = orb.detectAndCompute(img1_g,None)
            kp2, des2 = orb.detectAndCompute(img2_g,None)
            matches = matcher.match(des1, des2)
            matches = sorted(matches, key = lambda x : x.distance)

            points1 = np.array([kp1[m.queryIdx].pt for m in matches])
            points2 = np.array([kp2[m.trainIdx].pt for m in matches])

            E, mask = cv.findEssentialMat(points1, points2, self.K)
            val, R_k, t_k, mask, triangulatedPoints = cv.recoverPose(E, points1, points2, self.K, distanceThresh = 50, mask = mask)
            triangulatedPoints = triangulatedPoints[:3] / triangulatedPoints[3]

            R = R.dot(R_k)
            t = t + R_k.dot(t_k)
            if frame_path != self.frame_paths[1]:
                intersection, pre_idx, cur_idx = np.intersect1d(previous_points2, points1, assume_unique=True, return_indices=True)
                same = 0
                for i in range(len(pre_idx)):
                    if pre_idx[i] >= 0 and pre_idx[i] < len(previous_points2) and cur_idx[i] >= 0 and cur_idx[i] < len(points1):
                        if same == 0:
                            X_kminus1 = previous_triangulatedPoints[:, pre_idx[i]]
                            X_k = triangulatedPoints[:, cur_idx[i]]
                            same = same + 1
                        if same == 1:
                            X_prime_kminus1 = previous_triangulatedPoints[:, pre_idx[i]]
                            X_prime = triangulatedPoints[:, cur_idx[i]]
                            same = same + 1
                            break
                if same == 2:
                    t = t * previous_t_norm * norm(X_k - X_prime) / norm(X_kminus1 - X_prime_kminus1)

            previous_points2 = points2
            previous_triangulatedPoints = triangulatedPoints
            previous_t_norm = norm(t)
            queue.put((R, t))

            img2_ = img2
            for point in points2:
                cv.circle(img2_, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
            cv.imshow('frame', img2_)
            img1 = img2

            if cv.waitKey(30) == 27: break
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
