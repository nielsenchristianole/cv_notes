from typing import Optional
from itertools import product

import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def Pi(arr: np.ndarray) -> np.ndarray:
    """
    Also called Pi(x)

    Converts from homogeneous to inhomogeneous coordinates
    """
    return arr[:-1] / arr[-1]


def InvPi(arr: np.ndarray) -> np.ndarray:
    """
    Converts from inhomogeneous to homogeneous coordinates
    """
    return np.vstack((arr, np.ones(arr.shape[1])))


def camera_matrix(
    f: float=1.,
    delta: list[float]=(0,0),
    alpha: float=1.,
    beta: float=0.,
) -> np.ndarray:
    """
    f: focal length
    delta: principal point offset
    alpha: ratio of focal lengths f_y / f_x. Alpha!=1 means non-square pixels
    beta: skew factor with slope = beta / alpha. Beta!=0 means non-orthogonal axes

    Returns the camera matrix
    """
    return np.array([[f,  beta*f, delta[0]],
                     [0, alpha*f, delta[1]],
                     [0,       0,        1]])


def shortest_distance_to_line(ps: np.ndarray, l: np.ndarray) -> float:
    """
    Returns the shortest distance besteen each point in ps to a line l.
    """
    return abs(l.T @ ps) / (abs(ps[-1]) * np.linalg.norm(l[:-1]))



def box3d(n: int=16) -> np.ndarray:
    import itertools as it

    points = []
    N = tuple(np.linspace(-1, 1, n))
    
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i, ) * n, (j, ) * n, N])))
    
    return np.hstack(points) / 2

def projectpoints(
    K: np.ndarray,
    R: np.ndarray=np.eye(3),
    t: np.ndarray=np.zeros((3, 1)),
    Q: Optional[np.ndarray]=None,
    *,
    radial_dist_factors: Optional[list[float]]=None,
) -> np.ndarray:
    """
    K: Camera matrix
    R: Rotation matrix
    t: Translation vector
    Q: 3D points
    radial_dist_factors: the coefficients of the radial distortion polynomial
        dr(r) = k_3 * r**2 + k_5 * r**4 + k_7 * r**6 + ...
    """

    assert (Q is not None) or (radial_dist_factors is None), "Q must be provided if radial_dist_factors is provided"

    projection_matrix = K @ np.concatenate((R,t), axis=1)

    if Q is None:
        return projection_matrix
    
    if radial_dist_factors is not None:
        # regid transform
        _tmp = Pi(np.concatenate((R,t), axis=1) @ Q)
        
        # radial distortion
        _norm = np.linalg.norm(_tmp, axis=0)
        _scale = 1 + np.stack([c * _norm ** (2*i)
                            for i, c in enumerate(radial_dist_factors, start=1)]).sum(axis=0)
        _tmp = _tmp * _scale
        
        # camera projection
        return K @ InvPi(_tmp)

    return projection_matrix @ Q


def undistortImage(
    img: np.ndarray,
    radial_dist_factors: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    p = np.stack((x, y, np.ones_like(x))).reshape(3, -1)
    q = np.linalg.inv(K) @ p
    q_d = projectpoints(np.eye(3), Q=InvPi(q), radial_dist_factors=radial_dist_factors)
    p_d = K @ q_d
    x_d = p_d[0].reshape(x.shape).astype(np.float32)
    y_d = p_d[1].reshape(y.shape).astype(np.float32)
    assert (p_d[2]==1).all(), 'You did a mistake somewhere'
    img_undistorted = cv2.remap(img, x_d, y_d, cv2.INTER_LINEAR)
    return img_undistorted


def normalize2d(ps: np.ndarray) -> np.ndarray:
    """
    Creates a 2D normalization matrix
    where T_inv^-1 @ ps = ps_normalized
    """

    mu = ps.mean(axis=1)
    sigma = ps.std(axis=1)
    T_inv = np.array([[sigma[0], 0, mu[0]],
                  [0, sigma[1], mu[1]],
                  [0, 0, 1]])
    return T_inv


CrossOp = lambda v: np.array([[    0, -v[2],  v[1]],
                              [ v[2],     0, -v[0]],
                              [-v[1],  v[0],     0]])


def hest(
    ps: np.ndarray,
    qs: np.ndarray,
    *,
    normalize: bool=True
) -> np.ndarray:
    """
    Estimates the homography matrix between two sets of points where
    Pi(p) = Pi(H @ q)
    """
    ps = InvPi(ps)
    qs = InvPi(qs)

    if normalize:
        Tp_inv = normalize2d(ps)
        Tp = np.linalg.inv(Tp_inv)

        Tq_inv = normalize2d(qs)
        Tq = np.linalg.inv(Tq_inv)

        ps = Tp @ ps
        qs = Tq @ qs
    
    B = np.vstack([np.kron(q, CrossOp(p)) for p, q in zip(ps.T, qs.T)])
    _, _, V = np.linalg.svd(B)

    H_pred = V[-1].reshape(3, 3).T

    if normalize:
        return Tp_inv @ H_pred @ Tq
    return H_pred


def rotation_matrix(*thetas: list[float]) -> np.ndarray:
    """
    Returns a rotation matrix given the angles in the form of
    [theta_x, theta_y, theta_z]
    """
    if len(thetas) == 1:
        thetas = thetas[0]

    return Rotation.from_euler('xyz', thetas).as_matrix()


def triangulate(qs: list[np.ndarray], Ps: list[np.ndarray]) -> np.ndarray:
    """
    qs: list of 2d points shape (2, n)
    Ps: list of 3x4 projection matrices

    Triangulates points qs from two different views Ps
    """
    qs = [Pi(q) if q.shape[0] == 3 else q for q in qs]

    B = np.vstack([P[2][None, :] * q - P[:-1] for q, P in zip(qs, Ps)])
    _, _, V = np.linalg.svd(B)

    return V[-1][:, None]


prettyprint = lambda arr, pres=2: print(np.array2string(arr, precision=pres, suppress_small=True))


def pest(
    qs: np.ndarray,
    Qs: np.ndarray
) -> np.ndarray:
    """
    Estimates the projection matrix between two sets of points where
    Pi(P) = Pi(H @ Q)
    """
    qs = InvPi(qs)
    Qs = InvPi(Qs)
    
    B = np.vstack([np.kron(Q, CrossOp(q)) for q, Q in zip(qs.T, Qs.T)])
    _, _, V = np.linalg.svd(B)

    P_pred = V[-1].reshape(4, 3).T
    P_pred /= P_pred[2,2] # normalize

    return P_pred


def checkerboard_points(
    n: int=8,
    m: int=8,
    *,
    only_corners: bool=False
) -> np.ndarray:
    """
    Returns the 3D coordinates of a checkerboard
    """
    if only_corners:
        p = np.array(list(product([0, n-1], [0, m-1], [0.]))).T
    else:
        p = np.array(list(product(range(n), range(m), [0.]))).T

    bias = np.array([[(n-1)/2, (m-1)/2, 0]]).T
    return p - bias


def estimateHomographies(Q_omega: np.ndarray, qs: list[np.ndarray]) -> list[np.ndarray]:
    """
    Estimates the homographies between the 3D points and the 2D points.

    Q_omega: 3D points
    qs: 2D points
    """
    Q_omega_bar = Q_omega[:2]
    return [hest(q, Q_omega_bar) for q in qs]


def estimate_b(Hs: list[np.ndarray]) -> np.ndarray:
    """
    Estimates the vector b which is used for the estimation of the intrinsics
    """
    # b_true = np.array([B_true[0,0], B_true[0,1], B_true[1,1], B_true[0,2], B_true[1,2], B_true[2,2]])

    V = list()
    for H in Hs:

        v = lambda a, b: np.array([
            H[0,a]*H[0,b],
            H[0,a]*H[1,b] + H[1,a]*H[0,b],
            H[1,a]*H[1,b],
            H[2,a]*H[0,b] + H[0,a]*H[2,b],
            H[2,a]*H[1,b] + H[1,a]*H[2,b],
            H[2,a]*H[2,b]])

        # print(v(0, 0) @ b_true - H[:, 0].T @ B_true @ H[:, 0])

        v1 = v(0, 1)
        v2 = v(0, 0) - v(1, 1)        

        V.extend((v1, v2))
    V = np.stack(V)

    return np.linalg.svd(V)[-1][-1]


def estimateIntrinsics(Hs: list[np.ndarray]) -> np.ndarray:
    """
    Estimates the camera matrix K, from the homographies
    """
    b = estimate_b(Hs)

    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])
    
    v0 = (B[0,1]*B[0,2] - B[0,0]*B[1,2]) / (B[0,0]*B[1,1] - B[0,1]**2)
    lam = B[2,2] - (B[0,2]**2 + v0 * (B[0,1]*B[0,2] - B[0,0]*B[1,2])) / B[0,0]
    alpha = np.sqrt(lam / B[0,0])
    beta = np.sqrt(lam * B[0,0] / (B[0,0]*B[1,1] - B[0,1]**2))
    gamma = -B[0,1] * alpha**2 * beta / lam
    u0 = gamma * v0 / beta - B[0,2] * alpha**2 / lam

    K = np.array([[alpha, gamma, u0],
                  [    0,  beta, v0],
                  [    0,     0,  1]])

    return K


def estimateExtrinsics(K, Hs):
    """
    Estimate extrinsic parameters using Zhang's method for camera calibration.

    Args:
        K : 3x3 intrinsic matrix
        Hs : list of 3x3 homographies for each view
    """
    Kinv = np.linalg.inv(K)  # Inverse of the intrinsic matrix
    Rs = []
    ts = []
    for H in Hs:  # Extract columns from homography H
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]
        lambda_ = np.linalg.norm(Kinv @ h1)  # Normalize using the first column
        r1 = (Kinv @ h1) / lambda_  # Normalized first rotation vector
        r2 = (Kinv @ h2) / lambda_  # Normalized second rotation vector
        r3 = np.cross(r1, r2)  # Third rotation vector as cross product
        t = (Kinv @ h3 / lambda_).reshape(3, 1)  # Translation vector
        R = np.column_stack((r1, r2, r3)).T  # Construct rotation matrix
        Rs.append(R)
        ts.append(t)
    
    Hs = [H if t[2].item() > 0 else -H for t, H in zip(ts, Hs)]
    for H in Hs:  # Extract columns from homography H
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]
        lambda_ = np.linalg.norm(Kinv @ h1)  # Normalize using the first column
        r1 = (Kinv @ h1) / lambda_  # Normalized first rotation vector
        r2 = (Kinv @ h2) / lambda_  # Normalized second rotation vector
        r3 = np.cross(r1, r2)  # Third rotation vector as cross product
        t = (Kinv @ h3 / lambda_).reshape(3, 1)  # Translation vector
        R = np.column_stack((r1, r2, r3)).T  # Construct rotation matrix
        Rs.append(R)
        ts.append(t)

    Rs = np.array(Rs)  # Convert list to array for rotation matrices
    ts = np.array(ts)  # Convert list to array for translation vectors
    return Rs, ts

# Wrong implementation
if False:
    def estimateExtrinsics(K: np.ndarray, Hs: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Estimates the extrinsics given the camera matrix and the homographies

        K: Camera matrix
        Hs: Homographies

        Returns the rotation matrices and the translation vectors
        """
        Rs = list()
        ts = list()

        K_inv = np.linalg.inv(K)

        for H in Hs:
            t = K_inv @ H[:, 2]

            if t[2] < 0:
                t = -t
                H = -H

            r1 = K_inv @ H[:, 0][:, None]
            r2 = K_inv @ H[:, 1][:, None]

            r1 /= np.linalg.norm(r1, axis=0)
            r2 /= np.linalg.norm(r2, axis=0)
            r3 = np.cross(
                r1.squeeze(),
                r2.squeeze())[:, None]

            R = np.concatenate((r1, r2, r3), axis=1)
            Rs.append(R)
            ts.append(t[:, None])

        return Rs, ts


def calibratecamera(
    qs: list[np.ndarray],
    Q: np.ndarray,
    *,
    verbose: bool=False
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    qs: list of 2d points
    Qs: 3d points
    """
    Hs = estimateHomographies(Q, qs)
    K = estimateIntrinsics(Hs)
    Rs, ts = estimateExtrinsics(K, Hs)

    if verbose:
        err = 0
        num_plots = len(qs)
        fig, axs = plt.subplots(1, num_plots, figsize=(num_plots*5, 5))

        for R, t, q_true, ax in zip(Rs, ts, qs, axs.flatten()):
            q_est = Pi(projectpoints(K, R, t, InvPi(Q)))
            _err = np.linalg.norm(q_true - q_est, axis=0) ** 2
            err += _err.mean()

            ax.scatter(*q_true, label='True')
            ax.scatter(*q_est, label='Est')
            ax.legend()
        plt.show()

        print(f'Reprojection error: {err:.4}')

    return K, Rs, ts


def triangulate_nonlin(
    qs: list[np.ndarray],
    Ps: list[np.ndarray],
) -> np.ndarray:
    """
    Non-linear triangulation
    """
    import scipy.optimize

    def compute_residuals(Q):
        Q = Q[:, None]
        return np.concatenate(
            [Pi(P @ InvPi(Q)) - q
             for q, P in zip(qs, Ps)],
            axis=0
        ).mean(axis=1)
    
    x0 = Pi(triangulate(qs, Ps))[:,0]
    x = scipy.optimize.least_squares(compute_residuals, x0)
    return x.x[:, None]


def gaussian1DKernel(
    sigma: float,
    length: float=5,
    num: Optional[int]=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a 1D gaussian kernel and its derivative

    sigma: standard deviation
    length: how many standard deviations to consider
    num: number of samples
    """
    _s = sigma * length
    num = num or 2 * int(_s) + 1

    x = np.linspace(-_s, _s, num)
    g = np.exp(-x**2 / (2 * sigma**2))
    g /= g.sum()

    gd = -x / sigma**2 * g
    return g, gd


def gaussianSmoothing(
    im: np.ndarray,
    sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooths an image with a gaussian kernel, and returns the smoothed image and its derivatives

    im: image
    sigma: standard deviation
    """
    g, gd = gaussian1DKernel(sigma)

    im = cv2.sepFilter2D(im, -1, g, g)
    Ix = cv2.sepFilter2D(im, -1, gd, g)
    Iy = cv2.sepFilter2D(im, -1, g, gd)

    return im, Ix, Iy


def structureTensor(
    im: np.ndarray,
    sigma: float,
    epsilon: float
) -> np.ndarray:
    """
    Computes the structure tensor of an image

    im: image
    sigma: standard deviation for the gaussian smoothing
    epsilon: stadard deviation f
    """
    _, Ix, Iy = gaussianSmoothing(im, sigma)

    g_eps, _ = gaussian1DKernel(epsilon)

    a = cv2.sepFilter2D(Ix**2, -1, g_eps, g_eps)
    b = cv2.sepFilter2D(Iy**2, -1, g_eps, g_eps)
    c = cv2.sepFilter2D(Ix*Iy, -1, g_eps, g_eps)

    return a, b, c


def harrisMeasure(
    im: np.ndarray,
    sigma: float,
    epsilon: float,
    k: float=0.06
) -> np.ndarray:
    """
    Computes the Harris measure of an image
    """

    a, b, c = structureTensor(im, sigma, epsilon)

    det = a * b - c ** 2
    tr = a + b
    return det - k * tr ** 2


def cornerDetector(
    im: np.ndarray,
    sigma: float,
    epsilon: float,
    tau: float,
    k: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detects corners in an image

    Coordinates might be transposed
    """
    r = harrisMeasure(im, sigma, epsilon, k)

    _range = 0.1 * r.max(), 0.8 * r.max()
    if tau < _range[0] or tau > _range[1]:
        print(f"{tau=} should be in the range {_range}")

    c = (
        (r[1:-1, 1:-1] > tau) &
        (r[1:-1, 1:-1] > r[2:, 1:-1]) &
        (r[1:-1, 1:-1] > r[:-2, 1:-1]) &
        (r[1:-1, 1:-1] > r[1:-1, 2:]) &
        (r[1:-1, 1:-1] > r[1:-1, :-2]) &
        (r[1:-1, 1:-1] > r[2:, 2:]) &
        (r[1:-1, 1:-1] > r[:-2, :-2]) &
        (r[1:-1, 1:-1] > r[2:, :-2]) &
        (r[1:-1, 1:-1] > r[:-2, 2:])
    )
    c = np.pad(c, ((1, 1), (1, 1)))
    return np.where(c) 


def DrawLine(l, shape):
    #Checks where the line intersects the four sides of the image
    # and finds the two intersections that are within the frame
    def in_frame(l_im):
        q = np.cross(l.flatten(), l_im)
        q = q[:2]/q[2]
        if all(q>=0) and all(q+1<=shape[1::-1]):
            return q
    lines = [[1, 0, 0], [0, 1, 0], [1, 0, 1-shape[1]], [0, 1, 1-shape[0]]]
    P = [in_frame(l_im) for l_im in lines if in_frame(l_im) is not None]
    if (len(P)==0):
        print("Line is completely outside image")
    plt.plot(*np.array(P).T)


def test_points(n_in, n_out):
    """
    Samples random points with n_in is the number of inliners and n_out is the number of outliers
    """
    a = (np.random.rand(n_in) - .5) * 10
    b = np.vstack((a, a*.5+np.random.randn(n_in)*.25))
    points = np.hstack((b, 2*np.random.randn(2, n_out)))
    return np.random.permutation(points.T).T


def fit_line(q1, q2):
    """
    The line is parameterized as l = [a, b, c] where a*x + b*y + c = 0
    """
    a = np.cross(q1, q2)
    return a / np.linalg.norm(a[:2])


def pca_line(x): # assumes x is a (2 x n) array of points
    d = np.cov(x)[:, 0]
    d /= np.linalg.norm(d)
    l = [d[1], -d[0]]
    l.append(-(l@x.mean(1)))
    return np.array(l)


def distance_to_line(l, q):
    """
    Returns the distance between a line and a point

    The points are homogeneous 2D coordinates
    """
    l /= np.linalg.norm(l[:2])
    return abs(l.T @ q) / np.linalg.norm(l[:2])


def ransac_line(points, threshold=0.1, max_iters=1000, p=0.99):
    num_points = points.shape[1]

    sample = lambda: np.random.choice(num_points, 2, replace=False)

    best_inliers = 0
    best_line = None

    for i in range(1, max_iters+1):
        q1, q2 = points[:, sample()].T
        line = fit_line(q1, q2)
        inliers = distance_to_line(line, points) < threshold

        if inliers.sum() > best_inliers:
            best_inliers = inliers.sum()
            best_line = line
        
        eps_hat = 1 - best_inliers / num_points

        if np.log(1 - p) / np.log(1 - (1 - eps_hat)**i) < i:
            print(i)
            break

    inliers = distance_to_line(best_line, points) < threshold
    line = pca_line(Pi(points[:, inliers]))

    return line


def scaleSpaced(im: np.ndarray, sigma: float, n: int) -> np.ndarray:
    """
    Scales an image with a gaussian kernel

    im: image
    sigma: standard deviation
    n: number of scales
    """

    scales = [2**i for i in range(n)]
    scaled = np.zeros((im.shape[0], im.shape[1], n))

    for i, s in enumerate(scales):
        g, _ = gaussian1DKernel(sigma * s)
        scaled[:, :, i] = cv2.sepFilter2D(im, -1, g, g)

    return scaled


def differenceOfGaussians(im: np.ndarray, sigma: float, n: int) -> list[np.ndarray]:
    im_scales = scaleSpaced(im, sigma, n)
    return np.diff(im_scales, axis=2)


def detectBlobs(im: np.ndarray, sigma: float, n: int, tau) -> np.ndarray:
    """
    Detects blobs in an image
    """
    DoG = differenceOfGaussians(im, sigma, n)
    maxDoG = cv2.dilate(DoG, np.ones((3, 3)))

    sup = (
        (DoG > tau)     & # threshold
        (DoG == maxDoG))  # suppress non-maxima in channel
    sup[..., :-1] &= maxDoG[..., :-1] > maxDoG[..., 1:] # suppress non-maxima in smaller scale
    sup[..., 1:] &= maxDoG[..., 1:] > maxDoG[..., :-1] # suppress non-maxima in larger scale

    return np.where(sup)


def transformIm(img: np.ndarray, theta: float, s: float) -> np.ndarray:
    """"""
    center = np.array(img.shape[::-1]) / 2
    rot_mat = cv2.getRotationMatrix2D(center, theta, 1)

    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    p = np.stack((x, y, np.ones_like(x))).reshape(3, -1)

    p_d = rot_mat @ p
    p_d_mean = p_d.mean(axis=1, keepdims=True)
    p_d = (p_d - p_d_mean) * s + p_d_mean

    x_d = p_d[0].reshape(x.shape).astype(np.float32)
    y_d = p_d[1].reshape(y.shape).astype(np.float32)
    transformed = cv2.remap(img, x_d, y_d, cv2.INTER_LINEAR)
    return transformed


def Fest_8point(q1, q2):
    """
    Estimates the fundamental matrix using the 8-point algorithm
    """
    def B_i(q1, q2):
        x1, y1 = q1[0], q1[1]
        x2, y2 = q2[0], q2[1]
        return np.array([x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1])

    A = np.stack([B_i(_q1, _q2) for _q1, _q2 in zip(q1.T, q2.T)])
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    return F


def unpack_cv2_matches(
    kp1: tuple[cv2.KeyPoint, ...],
    kp2: tuple[cv2.KeyPoint, ...],
    matches: tuple[cv2.DMatch, ...]
) -> tuple[np.ndarray, np.ndarray]:
    
    q1 = np.array([kp1[m.queryIdx].pt for m in matches]).T
    q2 = np.array([kp2[m.trainIdx].pt for m in matches]).T
    return q1, q2


def SampsonsDistance(
    F: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray
) -> np.ndarray:
    """
    Sampson's distance
    """
    Fp1 = F @ p1
    p2F = p2.T @ F

    p2Fp1 = np.einsum('ij,ji->i', p2F, p1)
    denum = np.square(Fp1[:2]).sum(axis=0) + np.square(p2F[:, :2]).sum(axis=1)
    
    return np.square(p2Fp1) / denum


def ransac_F(
    q1: np.ndarray,
    q2: np.ndarray,
    t_sq=3.84*3**2,
    max_iters=200,
    *,
    return_match_filter: bool=False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    RANSAC for the fundamental matrix

    q1: points in image 1
    q2: points in image 2
    t_sq: the threshold for the distance squared (faster)
        The erros are assumed to be gaussian, thus the erros squared are chi-squared distributed.
        Setting the threshold to 3.84*sigma^2 will give a 95% confidence interval
    max_iters: the maximum number of iterations
    p: the probability of success
    """

    q1 = InvPi(q1)
    q2 = InvPi(q2)

    num_points = q1.shape[1]

    def get_sample() -> tuple[np.ndarray, np.ndarray]:
        idx = np.random.choice(num_points, 8, replace=False)
        return q1[:, idx], q2[:, idx]

    best_inliers = 0
    best_F = None

    for _ in range(1, max_iters+1):
        _q1, _q2 = get_sample()
        F = Fest_8point(_q1, _q2)

        inliers = SampsonsDistance(F, q1, q2) < t_sq

        if inliers.sum() > best_inliers:
            best_inliers = inliers.sum()
            best_F = F

    inliers = SampsonsDistance(best_F, q1, q2) < t_sq
    F = Fest_8point(q1[:, inliers], q2[:, inliers])

    if return_match_filter:
        inliers = SampsonsDistance(best_F, q1, q2) < t_sq
        return F, inliers

    return F


def homography_dist_approx(H, q1, q2):
    """
    Approximate the distance between the points and the homography
    """
    d1 = np.square(np.linalg.norm(q1 - Pi(H @ InvPi(q2)), axis=0))
    d2 = np.square(np.linalg.norm(q2 - Pi(np.linalg.inv(H) @ InvPi(q1)), axis=0))
    return d1 + d2


def ransac_H(
    q1: np.ndarray,
    q2: np.ndarray,
    t_sq=3.84*3**2,
    max_iters=200,
    *,
    return_match_filter: bool=False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    RANSAC for estimating a homography

    q1: points in image 1
    q2: points in image 2
    t_sq: the threshold for the distance squared (faster)
        The erros are assumed to be gaussian, thus the erros squared are chi-squared distributed.
        Setting the threshold to 3.84*sigma^2 will give a 95% confidence interval
    max_iters: the maximum number of iterations
    """

    num_points = q1.shape[1]

    def get_sample() -> tuple[np.ndarray, np.ndarray]:
        idx = np.random.choice(num_points, 8, replace=False)
        return q1[:, idx], q2[:, idx]

    best_inliers = 0
    best_H = None

    for _ in range(1, max_iters+1):
        _q1, _q2 = get_sample()
        H = hest(_q1, _q2)

        inliers = homography_dist_approx(H, q1, q2) < t_sq

        if inliers.sum() > best_inliers:
            best_inliers = inliers.sum()
            best_H = H

    inliers = homography_dist_approx(best_H, q1, q2) < t_sq
    H = hest(q1[:, inliers], q2[:, inliers])

    if return_match_filter:
        inliers = homography_dist_approx(best_H, q1, q2) < t_sq
        return H, inliers

    return H


def estHomographyRANSAC(
    kp1: tuple[cv2.KeyPoint, ...],
    des1: np.ndarray,
    kp2:tuple[cv2.KeyPoint, ...],
    des2: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Estimates the homography between two images using RANSAC

    args are the keypoints and the descriptors from SIFT
    kwargs are the arguments for ransac_H
    """
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(des1, des2)
    q1, q2 = unpack_cv2_matches(kp1, kp2, matches)
    return ransac_H(q1, q2, **kwargs)


def warpImage(
    im: np.ndarray,
    H: np.ndarray,
    xRange: tuple[int, int],
    yRange: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    T = np.eye(3)
    T[:2, 2] = [-xRange[0], -yRange[0]]
    H = T@H
    outSize = (xRange[1]-xRange[0], yRange[1]-yRange[0])
    mask = np.ones(im.shape[:2], dtype=np.uint8)*255
    imWarp = cv2.warpPerspective(im, H, outSize)
    maskWarp = cv2.warpPerspective(mask, H, outSize)
    return imWarp, maskWarp


def unwrap(
    imgs_primary: list[np.ndarray],
    imgs_secondary: list[np.ndarray],
    n_1: Optional[int]=None
) -> np.ndarray:
    n_1 = n_1 or 1

    fft_primary = np.fft.rfft(imgs_primary, axis=0)
    theta_primary = np.angle(fft_primary[1])

    fft_secondary = np.fft.rfft(imgs_secondary, axis=0)
    theta_secondary = np.angle(fft_secondary[1])

    # heterodyne principle
    theta_c = theta_primary - theta_secondary
    theta_c %= 2*np.pi

    o_primary = np.round((n_1 * theta_c - theta_primary) / (2 * np.pi))
    theta_est = (2 * np.pi * o_primary + theta_primary) / n_1

    return theta_est


def calculate_disparity(theta_est0: np.ndarray, theta_est1: np.ndarray, mask0: np.ndarray, mask1: np.ndarray) -> np.ndarray:

    disparity = np.zeros_like(theta_est0)
    for i0 in tqdm.trange(disparity.shape[0]):
        for j0 in range(disparity.shape[1]):
            if not mask0[i0, j0]:
                continue
            min_diff = np.inf
            for j1 in range(disparity.shape[1]):
                if not mask1[i0, j1]:
                    continue
                diff = abs(theta_est0[i0, j0] - theta_est1[i0, j1])
                if min_diff > diff:
                    min_diff = diff
                    disparity[i0, j0] = j0 - j1
    
    return disparity


