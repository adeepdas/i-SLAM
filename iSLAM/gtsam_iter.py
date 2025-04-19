import numpy as np
import gtsam
from typing import List, Tuple
from visual_odometry import extract_visual_odometry, rotate_frame
from imu_integration import integrate_imu_trajectory, read_imu_data
from visualization import animate_trajectory
import cv2
from bow import BoWMatcher

# create a new graph every time and optimize incrementally 

def gtsam_optimization(frames: List[dict], scale_factor: float = 1/6, vocab_path="vocab.pkl") -> np.ndarray:
    """
    Refine SLAM trajectory using GTSAM with visual odometry and IMU data.
    
    Args:
        frames (List[dict]): List of data frames
        
    Returns:
        refined_transforms (np.ndarray): Refined 4x4 transformation matrices
    """

    print("Performing visual odometry.")
    # Extract visual odometry poses
    vo_transforms = extract_visual_odometry(frames)
    #vo_transforms = np.eye(4)
    
    print("Performing IMU integration.")
    # Extract IMU poses
    timestamps, acc_data, gyro_data = read_imu_data(frames)
    imu_positions, imu_orientations = integrate_imu_trajectory(timestamps, gyro_data, acc_data)
    
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

    loop_noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    # 3) Initialize BoW matcher
    bow_matcher = BoWMatcher(vocab_path=vocab_path)
    bow_matcher.load_vocab()

    # 4) Keyframe DB
    keyframe_imgs  = []
    bow_database   = []
    keyframe_ids   = []

    initial_values = gtsam.Values()

    # for idx in range(0, len(vo_transforms)-1):
    for idx in range(0, 5):
        print("Loop 1 - Frame:", idx)

        # Create factor graph for this frame
        graph = gtsam.NonlinearFactorGraph()
        
        if idx == 0:
            # Add prior factor for first pose
            R = vo_transforms[idx, :3, :3]
            t = vo_transforms[idx, :3, -1]
            pose_vo = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))
            graph.add(gtsam.PriorFactorPose3(0, pose_vo, prior_noise))
            if not initial_values.exists(0):
                initial_values.insert(0, pose_vo)
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
            if not initial_values.exists(idx):
                initial_values.insert(idx, gtsam.Pose3())

        curr_img, _ = rotate_frame(frames[idx])
        bow_curr = bow_matcher.compute_bow_descriptor(curr_img)
        make_kf = False

        # Make Camera Intrinsics matrix
        sx = scale_factor
        sy = scale_factor
        # print(type(frames[idx]['fx']))
        K = np.array([[frames[idx]['fx'] * sx, 0, frames[idx]['cx'] * sx],
                      [0, frames[idx]['fy'] * sy, frames[idx]['cy'] * sy] ,
                      [0, 0, 1]])
        
        if idx == 0:
            make_kf = True

        else:
            sim_last = bow_matcher.compare_bow(bow_curr, bow_database[-1])
            if sim_last < 0.7:
                make_kf = True
        
        if make_kf:
            k_id = idx  # use idx as the keyframe ID
            keyframe_ids.append(k_id)
            keyframe_imgs.append(curr_img)
            bow_database.append(bow_curr)
            if not initial_values.exists(k_id):
                initial_values.insert(k_id, pose_vo)

            # c) Loop‐closure search against previous keyframes
            for k_i, bow_old in zip(keyframe_ids[:-1], bow_database[:-1]):
                sim = bow_matcher.compare_bow(bow_curr, bow_old)
                if sim > 0.8:
                    # geometric verification
                    pts_new, pts_old, _, _ = bow_matcher.match_features(curr_img, keyframe_imgs[keyframe_ids.index(k_i)])
                    E, mask = cv2.findEssentialMat(pts_new, pts_old, K, cv2.RANSAC, 0.999, 1.0)
                    _, R, t, mask_pose = cv2.recoverPose(E, pts_new, pts_old, K)
                    if mask_pose.sum() / len(mask_pose) > 0.3:
                        rel_pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t.flatten()))
                        graph.add(gtsam.BetweenFactorPose3(k_i, k_id, rel_pose, loop_noise))
                        print(f"  ↳ Loop closure between {k_i} ↔ {k_id}, sim={sim:.2f}")
                        break
        
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
    frames = np.load('data/old_unsync_data/rectangle_vertical.npy', allow_pickle=True)
    
    # Run batch optimization
    refined_transforms = gtsam_optimization(frames)
    
    # Visualize results
    orientations = refined_transforms[:, :3, :3]
    positions = refined_transforms[:, :3, -1]
    animate_trajectory(orientations, positions)