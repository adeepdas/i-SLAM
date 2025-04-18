import numpy as np
import sys
from typing import List, Tuple

# batch solution - optimizatize all at once

import gtsam
from visual_odometry import extract_visual_odometry
from imu_integration import integrate_imu_trajectory, read_imu_data
from visualization import animate_trajectory

def batch_optimization(frames: List[dict]) -> np.ndarray:
    """
    Perform batch optimization of SLAM trajectory using GTSAM with visual odometry and IMU data.
    
    Args:
        frames: List of frames containing RGB and depth images
        
    Returns:
        refined_transforms: Array of refined 4x4 transformation matrices
    """
    print("Performing visual odometry.")
    # Extract visual odometry poses
    vo_transforms = extract_visual_odometry(frames)
    
    print("Performing IMU integration.")
    # Extract IMU poses
    timestamps, acc_data, gyro_data = read_imu_data(frames)

    # integrate IMU data to get trajectory
    # set gravity to zero because iPhone does gravity compensation
    imu_positions, imu_orientations = integrate_imu_trajectory(timestamps, gyro_data, acc_data, g=np.zeros(3))
    
    # Create factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    
    # Create noise models - matching hw5_code.py gn_3d
    # Prior noise for first pose (same as in gn_3d)
    first_pose_prior_cov = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
    prior_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag(first_pose_prior_cov))
    
    # Visual odometry noise (using same structure as prior but with different values)
    vo_cov = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    vo_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag(vo_cov))
    
    # IMU noise (using same structure as prior but with different values)
    imu_cov = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    imu_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag(imu_cov))
    
    # Add prior factor for first pose
    pose = gtsam.Pose3()
    graph.add(gtsam.PriorFactorPose3(0, pose, prior_noise))
    initial_values.insert(0, pose)
    
    # Add factors for all poses
    print("Adding factors to graph.")
    for t in range(1, len(vo_transforms)):
        # Add visual odometry factor
        R_vo = vo_transforms[t, :3, :3]
        t_vo = vo_transforms[t, :3, -1]
        pose_vo = gtsam.Pose3(gtsam.Rot3(R_vo), gtsam.Point3(t_vo))
        graph.add(gtsam.BetweenFactorPose3(t-1, t, pose_vo, vo_noise))
        
        # Add IMU factor
        R_imu = imu_orientations[t]
        t_imu = imu_positions[t]
        pose_imu = gtsam.Pose3(gtsam.Rot3(R_imu), gtsam.Point3(t_imu))
        # graph.add(gtsam.BetweenFactorPose3(t-1, t, pose_imu, imu_noise))
        
        # Set initial value for optimization
        # make init pose from identity
        initial_values.insert(t, pose_imu)
    
    # Optimize
    print("Optimizing pose graph.")
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_values)
    result = optimizer.optimize()
    
    # Extract refined poses using GTSAM utility
    print("Extracting optimized poses.")
    refined_poses = gtsam.utilities.extractPose3(result)
    
    # Convert poses to transformation matrices
    refined_transforms = []
    for pose in refined_poses:
        R = pose[:-3].reshape(3, 3)
        t = pose[-3:]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, -1] = t
        refined_transforms.append(T)
    
    return np.array(refined_transforms)

if __name__ == "__main__":
    # Load data
    frames = np.load('data/rectangle_vertical.npy', allow_pickle=True)
    
    # Run batch optimization
    refined_transforms = batch_optimization(frames)
    
    # Visualize results
    orientations = refined_transforms[:, :3, :3]
    positions = refined_transforms[:, :3, -1]
    animate_trajectory(orientations, positions)