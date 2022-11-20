import open3d as o3d
import numpy as np
from numpy.linalg import norm, inv
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

        w = 640
        h = 360

        all_points = np.empty((0, 3), float)
        lines = np.empty((0, 2), np.int32)
        lines_color = np.empty((0, 3), float)

        origin = np.array([0, 0, 0, 1])
        corner_1 = np.append(inv(self.K).dot(np.array([0, 0, 1])), np.array([1]), axis=0)
        corner_2 = np.append(inv(self.K).dot(np.array([0, h-1, 1])), np.array([1]), axis=0)
        corner_3 = np.append(inv(self.K).dot(np.array([w-1, h-1, 1])), np.array([1]), axis=0)
        corner_4 = np.append(inv(self.K).dot(np.array([w-1, 0, 1])), np.array([1]), axis=0)

        i = 0
        
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    #TODO:
                    # insert new camera pose here using vis.add_geometry()
                    all_points = np.append(all_points, np.hstack((R, t)).dot(origin).reshape((1,3)), axis=0)
                    all_points = np.append(all_points, np.hstack((R, t)).dot(corner_1).reshape((1,3)), axis=0)
                    all_points = np.append(all_points, np.hstack((R, t)).dot(corner_2).reshape((1,3)), axis=0)
                    all_points = np.append(all_points, np.hstack((R, t)).dot(corner_3).reshape((1,3)), axis=0)
                    all_points = np.append(all_points, np.hstack((R, t)).dot(corner_4).reshape((1,3)), axis=0)
                    
                    lines = np.append(lines, np.array([[i, i + 1]]), axis=0)
                    lines_color = np.append(lines_color, np.array([[0, 0, 0]]), axis=0)

                    lines = np.append(lines, np.array([[i, i + 2]]), axis=0)
                    lines_color = np.append(lines_color, np.array([[0, 0, 0]]), axis=0)

                    lines = np.append(lines, np.array([[i, i + 3]]), axis=0)
                    lines_color = np.append(lines_color, np.array([[0, 0, 0]]), axis=0)

                    lines = np.append(lines, np.array([[i, i + 4]]), axis=0)
                    lines_color = np.append(lines_color, np.array([[0, 0, 0]]), axis=0)

                    lines = np.append(lines, np.array([[i + 1, i + 2]]), axis=0)
                    lines_color = np.append(lines_color, np.array([[0, 0, 1]]), axis=0)

                    lines = np.append(lines, np.array([[i + 2, i + 3]]), axis=0)
                    lines_color = np.append(lines_color, np.array([[0, 0, 1]]), axis=0)

                    lines = np.append(lines, np.array([[i + 3, i + 4]]), axis=0)
                    lines_color = np.append(lines_color, np.array([[0, 0, 1]]), axis=0)

                    lines = np.append(lines, np.array([[i + 4, i + 1]]), axis=0)
                    lines_color = np.append(lines_color, np.array([[0, 0, 1]]), axis=0)

                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(all_points)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector(lines_color)

                    vis.add_geometry(line_set)

                    i = i + 5

            except: pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def process_frames(self, queue):
        previous_R, previous_t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        orb = cv.ORB_create()
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        img1 = cv.imread(self.frame_paths[0])
        img1 = cv.undistort(img1, self.K, self.dist)
        img1_g = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

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
            
            E, mask = cv.findEssentialMat(points1, points2, self.K, threshold = 1)
            val, R, t, mask, triangulatedPoints = cv.recoverPose(E, points1, points2, self.K, distanceThresh = 50, mask = mask)
            triangulatedPoints = triangulatedPoints[:3] / triangulatedPoints[3]

            R = R.dot(previous_R)
            t = previous_t + R.dot(t)
            
            if frame_path != self.frame_paths[1]:
                pre_point1 = previous_R.dot(previous_triangulatedPoints[:, 0]) + previous_t.squeeze()
                pre_point2 = previous_R.dot(previous_triangulatedPoints[:, 1]) + previous_t.squeeze()
                cur_point1 = R.dot(previous_triangulatedPoints[:, 0]) + t.squeeze()
                cur_point2 = R.dot(previous_triangulatedPoints[:, 1]) + t.squeeze()
                scale = norm(previous_t) * norm(cur_point1 - cur_point2) / norm(pre_point1 - pre_point2) / norm(t)
                t_scale = scale * t
            
            else:
                t_scale = t

            queue.put((R, t_scale))
            
            img2_ = img2
            
            for point in points2:
                cv.circle(img2_, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
            cv.imshow('frame', img2_)
            
            previous_triangulatedPoints = triangulatedPoints
            previous_R = R
            previous_t = t
            img1_g = img2_g

            if cv.waitKey(30) == 27: break
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
