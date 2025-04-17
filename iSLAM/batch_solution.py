import numpy as np
import sys
from typing import List, Tuple

# batch solution - optimizatize all at once

import gtsam
from visual_odometry import extract_visual_odometry
from imu_integration import integrate_imu_trajectory
from visualization import animate_trajectory

def batch_optimization(frames: List[dict], imu_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform batch optimization of SLAM trajectory using GTSAM with visual odometry and IMU data.
    
    Args:
        frames: List of frames containing RGB and depth images
        imu_file: Path to IMU data file
        
    Returns:
        refined_transforms: Array of refined 4x4 transformation matrices
        refined_positions: Array of refined 3D positions
        
    """

    print(len(frames))

    # Extract visual odometry poses
    vo_transforms, vo_positions = extract_visual_odometry(frames)
    
    # Extract IMU poses
    imu_positions, imu_orientations = integrate_imu_trajectory(imu_file)
    
    # Create factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    
    # Create noise models
    # Visual odometry noise (higher uncertainty)
    vo_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    # IMU noise (lower uncertainty)
    imu_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]))
    # Prior noise for first pose
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]))
    
    # Add prior factor for first pose
    R = vo_transforms[0, :3, :3]
    t = vo_transforms[0, :3, 3]
    pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))
    graph.add(gtsam.PriorFactorPose3(0, pose, prior_noise))
    initial_values.insert(0, pose)
    
    # Add factors for all poses
    for t in range(1, len(vo_transforms[0])):
        print("Batch Optimization 1 - Frame:", t)
        # Add visual odometry factor
        R_vo = vo_transforms[t, :3, :3]
        t_vo = vo_transforms[t, :3, 3]
        pose_vo = gtsam.Pose3(gtsam.Rot3(R_vo), gtsam.Point3(t_vo))
        graph.add(gtsam.BetweenFactorPose3(t-1, t, pose_vo, vo_noise))
        
        # Add IMU factor
        R_imu = imu_orientations[t]
        t_imu = imu_positions[t]
        pose_imu = gtsam.Pose3(gtsam.Rot3(R_imu), gtsam.Point3(t_imu))
        graph.add(gtsam.BetweenFactorPose3(t-1, t, pose_imu, imu_noise))
        
        # Set initial value
        initial_values.insert(t, pose_vo)
    
    # Create optimizer parameters
    parameters = gtsam.GaussNewtonParams()
    parameters.setVerbosity("TERMINATION")  # Print optimization progress
    parameters.setMaxIterations(100)  # Maximum number of iterations
    
    # Create optimizer
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_values, parameters)
    
    # Optimize
    result = optimizer.optimize()
    
    # Extract refined poses
    refined_transforms = []
    for t in range(len(frames)):
        print("Batch Optimization 2 - Frame:", t)
        pose = result.atPose3(t)
        R = pose.rotation().matrix()
        t = pose.translation().vector()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        refined_transforms.append(T)
    
    refined_transforms = np.array(refined_transforms)
    refined_positions = refined_transforms[:, :3, 3]
    
    return refined_transforms, refined_positions

if __name__ == "__main__":
    # Load data
    frames = np.load('data/rectangle_vertical.npy', allow_pickle=True)
    imu_file = 'data/rectangle_vertical.npy'  # Assuming IMU data is in the same file
    
    # Run batch optimization
    refined_transforms, refined_positions = batch_optimization(frames, imu_file)
    
    # Visualize results
    orientations = refined_transforms[:, :3, :3]
    animate_trajectory(orientations, refined_positions)
