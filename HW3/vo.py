import open3d as o3d
import numpy as np
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
        for frame_path in self.frame_paths[1:]:
            img = cv.imread(frame_path)
            #TODO: compute camera pose here
            
            queue.put((R, t))
             
            cv.imshow('frame', img)
            if cv.waitKey(30) == 27: break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
