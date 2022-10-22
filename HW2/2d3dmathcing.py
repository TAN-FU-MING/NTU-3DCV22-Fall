from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polycompanion
from numpy.linalg import eig, norm
import random
import cv2


images_df = pd.read_pickle("data/images.pkl")
train_df = pd.read_pickle("data/train.pkl")
points3D_df = pd.read_pickle("data/points3D.pkl")
point_desc_df = pd.read_pickle("data/point_desc.pkl")

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    return solveP3PRansac(points3D, points2D, cameraMatrix, distCoeffs)

def solveP3PRansac(points3D, points2D, cameraMatrix, distCoeffs):
    N = 100
    for i in range(N):
        chosen_idx = random.sample(range(len(points3D)), 3)
        chosen_points3D = np.array([points3D[i] for i in chosen_idx])
        chosen_points2D = np.array([points2D[i] for i in chosen_idx])
        rvec, tvec = solveP3P(chosen_points3D, chosen_points2D, cameraMatrix, distCoeffs)

        # Distortion

    return rvec, tvec, inliers

def solveP3P(points3D, points2D, cameraMatrix, distCoeffs):

    x = points3D.T
    x1, x2, x3 = points3D[0], points3D[1], points3D[2]
    X = [x1, x2, x3]

    u = np.concatenate((points2D.T, np.ones((1, points2D.shape[0]))))

    v = np.linalg.inv(cameraMatrix).dot(u)
    v1, v2, v3 = v.T[0], v.T[1], v.T[2]
    V = [v1, v2, v3]

    C_ab, C_ac, C_bc = distance.cosine(v1, v2), distance.cosine(v1, v3), distance.cosine(v2, v3)
    R_ab, R_ac, R_bc = distance.euclidean(x1, x2), distance.euclidean(x1, x3), distance.euclidean(x2, x3)

    K1, K2 = (R_bc / R_ac) ** 2, (R_bc / R_ab) ** 2

    G4 = (K1 * K2 - K1 - K2) ** 2 - 4 * K1 * K2 * C_bc ** 2
    G3 = 4 * (K1 * K2 - K1 - K2) * K2 * (1 - K1) * C_ab + 4 * K1 * C_bc * ((K1 * K2 - K1 + K2) * C_ac + 2 * K2 * C_ab * C_bc)
    G2 = (2 * K2 * (1 - K1) * C_ab) ** 2 + 2 * (K1 * K2 - K1 - K2) * (K1 * K2 + K1 - K2) + 4 * K1 * ((K1 - K2) * C_bc ** 2 + K1 * (1 - K2) * C_ac ** 2 - 2 * (1 + K1) * K2 * C_ab * C_ac * C_bc)
    G1 = 4 * (K1 * K2 + K1 - K2) * K2 * (1 - K1) * C_ab + 4 * K1 * ((K1 * K2 - K1 + K2) * C_ac * C_bc + 2 * K1 * K2 * C_ab * C_ac ** 2)
    G0 = (K1 * K2 + K1 - K2) ** 2 - 4 * K1**2 * K2 * C_ac ** 2

    companion = polycompanion([G0, G1, G2, G3, G4])
    roots = eig(companion)[0]
    R_list = []
    for root in roots:
        a = ((R_ab ** 2) / (root ** 2 - 2 * root * C_ab + 1)  ) ** 0.5
        a_list = [a, -a]

        for a in a_list:
            m, p, q = 1 - K1, 2 * (K1 * C_ac - root * C_bc), root ** 2 - K1
            m_prime, p_prime, q_prime = 1, 2 * (-root * C_bc), (root ** 2 * (1 - K2) + 2 * root * K2 * C_ab - K2)

            y = -(m_prime * q - m * q_prime) / (p * m_prime - p_prime * m)
            b = root * a
            c = y * a

            T = trilateration(x1, x2, x3, a, b, c)
            
            for t in T:
                lambda_v_list, xt_list = np.array([]), np.array([])

                for i in range(len(X)):
                    lambda_ = norm(X[i] - t) / norm(V[i])
                    xt = X[i] - t
                    lambda_v_list = np.append(lambda_v_list, lambda_ * V[i])
                    xt_list = np.append(xt_list, xt)
                
                lambda_v = lambda_v_list.reshape((3, 3)).T
                xt = xt_list.reshape((3, 3)).T
                R = lambda_v.dot(np.linalg.inv(xt))
                if (np.linalg.det(R) < 0):
                    R = -R
                R_list.append(R)
    R_list = np.array(R_list)
    print(len(R_list))
    for i in range(len(R_list)):
        print(np.linalg.det(R_list[i]))
    return rvec, tvec

def trilateration(P1, P2, P3, r1, r2, r3):

    p1 = np.array([0, 0, 0])
    p2 = np.array([P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2]])
    p3 = np.array([P3[0] - P1[0], P3[1] - P1[1], P3[2] - P1[2]])

    v1 = p2 - p1
    v2 = p3 - p1

    Xn = v1 / norm(v1)

    tmp = np.cross(v1, v2)

    Zn = tmp / norm(tmp)

    Yn = np.cross(Xn, Zn)

    i = np.dot(Xn, v2)
    d = np.dot(Xn, v1)
    j = np.dot(Yn, v2)

    X = ((r1 ** 2) - (r2 ** 2) + (d ** 2)) / (2 * d)
    Y = (((r1 ** 2) - (r3 ** 2) + (i ** 2) + (j ** 2)) / (2 * j)) - ((i / j) * (X))
    Z1 = np.sqrt(max(0, r1 ** 2 - X ** 2 - Y ** 2))
    Z2 = -Z1 

    K1 = P1 + X * Xn + Y * Yn + Z1 * Zn
    K2 = p1 + X * Xn + Y * Yn - Z2 * Zn

    return K1, K2

# Process model descriptors
desc_df = average_desc(train_df, points3D_df)
kp_model = np.array(desc_df["XYZ"].to_list())
desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

# Load query image
idx = 200
fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

# Load query keypoints and descriptors
points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==idx]
kp_query = np.array(points["XY"].to_list())
desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

# Find correspondance and solve pnp
rvec, tvec, inliers = pnpsolver((kp_query, desc_query),(kp_model, desc_model))
rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat()
tvec = tvec.reshape(1,3)

# Get camera pose groudtruth 
ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
tvec_gt = ground_truth[["TX","TY","TZ"]].values

