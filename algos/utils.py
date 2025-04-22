import numpy as np


def skew(vec):
    """
    Converts a 3-vector into a 3x3 skew-symmetric matrix.
    
    Args:
        vec (np.ndarray): vector of shape (3,)
    
    Returns:
        hat (np.ndarray): matrix of shape (3, 3)
    """
    return np.array([
        [0,      -vec[2], vec[1]],
        [vec[2],  0,     -vec[0]],
        [-vec[1], vec[0],  0]
    ])

def hat(xi):
    """
    Converts a 6D twist vector to 4x4 SE(3) matrix.
    
    Args:
        xi (np.ndarray): twist vector of shape (6,)
    
    Returns:
        hat (np.ndarray): matrix of shape (4, 4)
    """
    omega = xi[0:3]
    v = xi[3:6]
    mat = np.zeros((4, 4))
    mat[:3, :3] = skew(omega)
    mat[:3, 3] = v
    return mat

def transform(T, pts):
    """
    Apply SE(3) transform T to 3D points (Nx3)

    Args:
        T (np.ndarray): SE(3) matrix of shape (4, 4)
        pts (np.ndarray): points of shape (N, 3)
    
    Returns:
        pts_transformed (np.ndarray): transformed points of shape (N, 3)
    """
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))  # Nx4
    return (T @ pts_h.T).T[:, :3]

def to_transform(R, t):
    """
    Convert rotation matrix and translation vector to SE(3) matrix.

    Args:
        R (np.ndarray): rotation matrix of shape (3, 3)
        t (np.ndarray): translation vector of shape (3,)

    Returns:
        T (np.ndarray): SE(3) matrix of shape (4, 4)
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T