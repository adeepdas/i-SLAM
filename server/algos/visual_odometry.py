import orb as orb
import fit_transform3D as fit_transform3D
from visualization import plot_trajectory, animate_trajectory
import numpy as np
import cv2
from typing import List
import argparse


def depth_to_pointcloud(depth, fx, fy, cx, cy):
    """
    Args:
        depth (np.ndarray): Depth image of shape (H, W)
        fx (float): Focal length in the x direction
        fy (float): Focal length in the y direction
        cx (float): Principal point in the x direction
        cy (float): Principal point in the y direction

    Returns:
        points (np.ndarray): Point cloud of shape (H, W, 3)
    """
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    Z = depth
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    points = np.stack((X, Y, -Z), axis=-1)

    return points

def rotate_frame(frame):
    """
    Rotate the frame by 90 degrees counterclockwise.
    """
    bgr = cv2.rotate(frame['bgr'], cv2.ROTATE_90_CLOCKWISE)
    depth = cv2.rotate(frame['depth'], cv2.ROTATE_90_CLOCKWISE)
    return bgr, depth

def extract_visual_odometry(frames: List[dict], 
                            min_matches: int = 100,
                            scale_factor: float = 1/6,
                            depth_threshold: float = -20,
                            ransac_threshold: float = 0.05,
                            ransac_iterations: int = 500):
    """
    Extract relative poses from visual odometry using feature matching and RANSAC.
    
    Args:
        frames (List[dict]): List of frames containing RGB and depth images with camera parameters
        min_matches (int): Minimum number of feature matches required to estimate transform
        scale_factor (float): Factor to scale down image resolution (default: 1/6)
        depth_threshold (float): Maximum depth value to consider (default: -20)
        ransac_threshold (float): RANSAC inlier threshold (default: 0.05)
        ransac_iterations (int): Maximum number of RANSAC iterations (default: 500)
        
    Returns:
        timestamps (np.ndarray): Timestamps of shape (N,)
        transforms (np.ndarray): Transformation matrices of shape (N, 4, 4)

    Raises:
        RuntimeError: If insufficient feature matches are found
    """ 
    timestamps = [0.]
    transforms = [np.eye(4)]
    for t in range(len(frames)-1):
        frame_prev = frames[t]
        frame_curr = frames[t+1]
        
        # rotate frames to correct orientation
        bgr_prev, depth_prev = rotate_frame(frame_prev)
        bgr_curr, depth_curr = rotate_frame(frame_curr)
        
        # extract and match features
        matches_prev, matches_curr = orb.feature_extraction(bgr_prev, bgr_curr)
        if len(matches_prev) < min_matches:
            print(f"Warning: Insufficient matches at frame {t} ({len(matches_prev)} < {min_matches})")
            continue
        
        # convert depth to point clouds
        sx = scale_factor
        sy = scale_factor
        pc_prev = depth_to_pointcloud(
            depth_prev,
            frame_prev['fx'] * sx,
            frame_prev['fy'] * sy,
            frame_prev['cx'] * sx,
            frame_prev['cy'] * sy
        )
        pc_curr = depth_to_pointcloud(
            depth_curr,
            frame_curr['fx'] * sx,
            frame_curr['fy'] * sy,
            frame_curr['cx'] * sx,
            frame_curr['cy'] * sy
        )
        
        # extract 3D points for matched features
        P = pc_prev[matches_prev[:, 1], matches_prev[:, 0], :]
        Q = pc_curr[matches_curr[:, 1], matches_curr[:, 0], :]
        
        # filter invalid points
        valid_P = (P[:, 2] != 0) & (P[:, 2] > depth_threshold)
        valid_Q = (Q[:, 2] != 0) & (Q[:, 2] > depth_threshold)
        # must filter both point clouds using the same indices
        valid_idx = valid_P & valid_Q
        P = P[valid_idx]
        Q = Q[valid_idx]
        
        # estimate transform using RANSAC
        T = fit_transform3D.ransac(Q, P, threshold=ransac_threshold, max_iterations=ransac_iterations)
        transforms.append(T)
        timestamps.append(frame_curr['timestamp'])
    
    timestamps = np.array(timestamps)
    transforms = np.array(transforms)
    
    return timestamps, transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract visual odometry from video data')
    parser.add_argument('--input', type=str, default='output/video_data_test.npy',
                      help='Path to input video data file')
    args = parser.parse_args()

    np.random.seed(42) 

    frames = np.load(args.input, allow_pickle=True)
    
    # extract visual odometry poses
    _, transforms = extract_visual_odometry(frames)

    # convert relative poses to absolute poses
    for i in range(1, len(transforms)):
        transforms[i] = transforms[i-1] @ transforms[i]
    
    # visualize trajectory
    orientations = transforms[:, :3, :3]
    positions = transforms[:, :3, -1]
    # animate_trajectory(orientations, positions)
    plot_trajectory(orientations, positions)