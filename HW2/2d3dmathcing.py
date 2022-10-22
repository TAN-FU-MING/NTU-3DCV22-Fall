from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polycompanion
from numpy.linalg import eig, norm, svd, inv
import random
import cv2
import open3d as o3d
from tqdm import trange
random.seed(2022)

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

    return Ransac(points3D, points2D, cameraMatrix, distCoeffs)

def Ransac(points3D, points2D, cameraMatrix, distCoeffs):
    N = 20
    threshold = 15
    k = 6

    num_best_inliners = 0

    for i in range(N):
        chosen_idx = random.sample(range(len(points3D)), k)
        chosen_points3D = np.array([points3D[i] for i in chosen_idx])
        chosen_points2D = np.array([points2D[i] for i in chosen_idx])
        # R, t = solveP3P(chosen_points3D, chosen_points2D, cameraMatrix, distCoeffs)
        R_, t = solveDLT(chosen_points3D, chosen_points2D, cameraMatrix, distCoeffs)

        M = cameraMatrix.dot(np.concatenate((R_, t), axis=1))
        u_homo = M.dot(np.concatenate((points3D.T, np.ones((1, points3D.shape[0])))))
        u = (u_homo / u_homo[2, :])[:2].T
        error_vector, error = error_computation(points2D, u)

        idx = np.where(error_vector < threshold)[0]
        num_inliers = len(idx)
        
        if num_inliers > num_best_inliners:
            num_best_inliners = num_inliers
            best_R = R_
            best_T = t
    r = R.from_matrix(best_R)
    return r.as_rotvec(), best_T.T, num_best_inliners

def error_computation(p_t, p_t_hat):
    diff = (p_t - p_t_hat)**2
    diff = diff.sum(axis = 1)
    error_vector = np.sqrt(diff)
    error = error_vector.sum()/p_t.shape[0]

    return error_vector, error

def solveDLT(points3D, points2D, cameraMatrix, distCoeffs):
    A = []
    for i in range(len(points3D)):
        [X, Y, Z] = points3D[i]
        [u, v] = points2D[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    U, S, VH = svd(A)
    F1 = np.array([[VH[-1, 0], VH[-1, 1], VH[-1, 2]],
                   [VH[-1, 4], VH[-1, 5], VH[-1, 6]],
                   [VH[-1, 8],VH[-1, 9], VH[-1, 10]]])
    F2 = np.array([[VH[-1, 3]], [VH[-1, 7]], [VH[-1, 11]]])
    R_ = inv(cameraMatrix).dot(F1)
    t = inv(cameraMatrix).dot(F2)
    return R_, t

def solveP3P(points3D, points2D, cameraMatrix, distCoeffs):
    x = points3D.T
    x1, x2, x3 = points3D[0], points3D[1], points3D[2]
    X = [x1, x2, x3]

    u = np.concatenate((points2D.T, np.ones((1, points2D.shape[0]))))
    
    v = inv(cameraMatrix).dot(u)
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
    R_list, T_list = [], []

    for root in roots:
        if (isinstance(root, complex)):
            continue
        a = ((R_ab ** 2) / (root ** 2 - 2 * root * C_ab + 1)) ** 0.5
        a_list = [a, -a]
        m, p, q = 1 - K1, 2 * (K1 * C_ac - root * C_bc), root ** 2 - K1
        m_prime, p_prime, q_prime = 1, 2 * (-root * C_bc), (root ** 2 * (1 - K2) + 2 * root * K2 * C_ab - K2)
        
        for a in a_list:
            
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
                R = lambda_v.dot(inv(xt))
                if (det(R) > 0):
                    R_list.append(R)
                    T_list.append(t)
    
    return np.array(R_list), np.array(T_list)

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

R_list, T_list = [], []
R_diff_list, T_diff_list = [], []

for idx in trange(1, point_desc_df['IMAGE_ID'].max()):
    # Load query image
    # fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
    # rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

    # Load query keypoints and descriptors
    points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==idx]
    kp_query = np.array(points["XY"].to_list())
    desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

    # Find correspondance and solve pnp
    rvec, tvec, inliers = pnpsolver((kp_query, desc_query),(kp_model, desc_model))
    r = R.from_rotvec(rvec.reshape(1, 3))
    rotq = r.as_quat()
    tvec = tvec.reshape(1, 3)
    
    R_list.append(r.as_matrix()[0])
    T_list.append(tvec)

    # Get camera pose groudtruth 
    ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
    rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
    tvec_gt = ground_truth[["TX","TY","TZ"]].values
    
    temp = np.abs(rotq_gt.dot(rotq.T))
    if temp < 1:
        R_diff = 2 * np.arccos(temp)
    else:
        R_diff = 2 * np.arccos(1)

    T_diff = distance.euclidean(tvec_gt[0], tvec[0])

    R_diff_list.append(R_diff)
    T_diff_list.append(T_diff)

print("Median of relative rotation angle differences:", np.median(R_diff_list))
print("Median of translation differences:", np.median(T_diff_list))

R_list, T_list = np.array(R_list), np.array(T_list)
M_inv = []

for r, t in zip(R_list, T_list):
    t = np.array(t).T
    rt = np.concatenate((r, t), axis=1)
    M = np.concatenate((rt, [[0, 0, 0, 1]]), axis = 0)
    M_inv.append(inv(M))

apex, base = [], []
for m in M_inv:
    apex.append(m.dot([0, 0, 0, 1]).reshape((1,4)))
    base.append(m.dot([0.2, 0.2, 1, 1]).reshape((1,4)))
    base.append(m.dot([0.2, -0.2, 1, 1]).reshape((1,4)))
    base.append(m.dot([-0.2, 0.2, 1, 1]).reshape((1,4)))
    base.append(m.dot([-0.2, -0.2, 1, 1]).reshape((1,4)))

lines, lines_color = [], []
for i in range(len(apex)-2):
    lines.append([i, i+1])
    lines_color.append([0, 1, 0])
apex, lines, lines_color = np.array(apex).reshape(-1, 4), np.array(lines).reshape(-1, 2), np.array(lines_color).reshape(-1, 3)

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(apex[:, :-1])
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(lines_color)

triangles = []
for i in range(0, len(base), 4):
    triangles.append([i, i+1, i+2])
    triangles.append([i+2, i+3, i])

base, triangles = np.array(base).reshape(-1, 4), np.array(triangles).reshape(-1, 3)
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(base[:,:-1])
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.paint_uniform_color([0, 0, 1])

points3D_arr = np.array(points3D_df["XYZ"].to_list())
points3D_RGB_arr = np.array(points3D_df["RGB"].to_list()) / 255
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points3D_arr)
point_cloud.colors = o3d.utility.Vector3dVector(points3D_RGB_arr)

o3d.visualization.draw_geometries([line_set, mesh, point_cloud])