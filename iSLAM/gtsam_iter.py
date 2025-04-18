import numpy as np
import gtsam
from typing import List, Tuple
from visual_odometry import extract_visual_odometry
from imu_integration import integrate_imu_trajectory
from visualization import animate_trajectory

# create a new graph every time and optimize incrementally 

def gtsam_optimization(frames: List[dict], imu_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine SLAM trajectory using GTSAM with visual odometry and IMU data.
    
    Args:
        frames: List of frames containing RGB and depth images
        imu_file: Path to IMU data file
        
    Returns:
        refined_transforms: Array of refined 4x4 transformation matrices
        refined_positions: Array of refined 3D positions
    """

    print("Performing visual odometry.")
    # Extract visual odometry poses
    vo_transforms, vo_positions = extract_visual_odometry(frames)
    
    print("Performing IMU integration.")
    # Extract IMU poses
    imu_positions, imu_orientations = integrate_imu_trajectory(imu_file)
    
    # Create ISAM2 optimizer
    isam = gtsam.ISAM2()
    
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
    
    for idx in range(0, len(vo_transforms)-1):
        print("Loop 1 - Frame:", idx)

        # Create factor graph for this frame
        graph = gtsam.NonlinearFactorGraph()
        initial_values = gtsam.Values()
        
        if idx == 0:
            # Add prior factor for first pose
            R = vo_transforms[idx, :3, :3]
            t = vo_transforms[idx, :3, -1]
            pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))
            graph.add(gtsam.PriorFactorPose3(0, pose, prior_noise))
            initial_values.insert(0, pose)
        else:
            # Add visual odometry factor
            R_vo = vo_transforms[idx, :3, :3]
            t_vo = vo_transforms[idx, :3, -1]
            pose_vo = gtsam.Pose3(gtsam.Rot3(R_vo), gtsam.Point3(t_vo))
            graph.add(gtsam.BetweenFactorPose3(idx-1, idx, pose_vo, vo_noise))
            
            # Add IMU factor
            R_imu = imu_orientations[idx]
            t_imu = imu_positions[idx]
            pose_imu = gtsam.Pose3(gtsam.Rot3(R_imu), gtsam.Point3(t_imu))
            graph.add(gtsam.BetweenFactorPose3(idx-1, idx, pose_imu, imu_noise))
            
            # Set initial value
            initial_values.insert(idx, gtsam.Pose3())
        
        # Update ISAM2
        isam.update(graph, initial_values)
        result = isam.calculateEstimate()

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
        T[:3, 3] = t
        refined_transforms.append(T)
    
    return np.array(refined_transforms)

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    # Load data
    frames = np.load('data/basement_rectangle_2.npy', allow_pickle=True)
    imu_file = 'data/basement_rectangle_2.npy'  # Assuming IMU data is in the same file
    
    # Run batch optimization
    refined_transforms = gtsam_optimization(frames, imu_file)
    
    # Visualize results
    orientations = refined_transforms[:, :3, :3]
    positions = refined_transforms[:, :3, 3]
    animate_trajectory(orientations, positions)
