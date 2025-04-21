import numpy as np
import cv2
import matplotlib.pyplot as plt
from iLoco.visual_odometry import extract_visual_odometry, depth_to_pointcloud, rotate_frame
from iLoco.orb import feature_extraction
from iLoco.gtsam_iter import graph_optimization
from iLoco.visualization import plot_trajectory

def visualize_results(video_data, imu_data, frame_idx=-1):
    """
    Visualize the RGB frame, depth frame, feature matches, pointcloud, and trajectory plot in a grid layout.
    
    Args:
        video_data (dict): RGBD video data
        imu_data (dict): IMU data
        frame_idx (int): Index of the frame to visualize. Defaults to -1 (last frame)
    """
    # Get the frame to visualize
    frame = video_data[frame_idx]
    prev_frame = video_data[frame_idx-1] if frame_idx > 0 else frame
    
    # Rotate frames to correct orientation
    bgr_prev, depth_prev = rotate_frame(prev_frame)
    bgr_curr, depth_curr = rotate_frame(frame)
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Top left: RGB and Depth frames side by side
    ax1 = plt.subplot(221)
    depth_map = depth_curr
    depth_map[np.isnan(depth_map)] = 0
    depth_map[np.isinf(depth_map)] = 0
    depth_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
    depth_8bit = depth_normalized.astype(np.uint8)
    colored_depth = cv2.applyColorMap(255-depth_8bit, cv2.COLORMAP_JET)
    
    # Stack RGB and depth frames horizontally
    combined_frames = np.hstack((bgr_curr, colored_depth))
    plt.imshow(cv2.cvtColor(combined_frames, cv2.COLOR_BGR2RGB))
    plt.title('RGB (Left) and Depth (Right) Frames')
    plt.axis('off')
    
    # 2. Top right: Feature matches
    ax2 = plt.subplot(222)
    matches_prev, matches_curr = feature_extraction(bgr_prev, bgr_curr)
    
    # Create visualization of matches
    img_combined = np.hstack((bgr_prev, bgr_curr))
    for pt1, pt2 in zip(matches_prev, matches_curr):
        pt2_offset = (int(pt2[0] + bgr_prev.shape[1]), int(pt2[1]))
        cv2.line(img_combined, (int(pt1[0]), int(pt1[1])), pt2_offset, (0, 255, 0), 1)
    
    plt.imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matches')
    plt.axis('off')
    
    # 3. Bottom left: Pointcloud of matches
    ax3 = plt.subplot(223, projection='3d')
    
    # Convert depth to pointcloud
    scale_factor = 1/6
    pc_prev = depth_to_pointcloud(
        depth_prev,
        prev_frame['fx'] * scale_factor,
        prev_frame['fy'] * scale_factor,
        prev_frame['cx'] * scale_factor,
        prev_frame['cy'] * scale_factor
    )
    pc_curr = depth_to_pointcloud(
        depth_curr,
        frame['fx'] * scale_factor,
        frame['fy'] * scale_factor,
        frame['cx'] * scale_factor,
        frame['cy'] * scale_factor
    )
    
    # Extract 3D points for matched features
    P = pc_prev[matches_prev[:, 1], matches_prev[:, 0], :]
    Q = pc_curr[matches_curr[:, 1], matches_curr[:, 0], :]
    
    # Filter invalid points
    valid_P = (P[:, 2] != 0) & (P[:, 2] > -20)
    valid_Q = (Q[:, 2] != 0) & (Q[:, 2] > -20)
    valid_idx = valid_P & valid_Q
    P = P[valid_idx]
    Q = Q[valid_idx]
    
    # Plot pointclouds
    ax3.scatter(P[:, 0], P[:, 1], P[:, 2], c='r', marker='o', label='Previous Frame')
    ax3.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c='b', marker='o', label='Current Frame')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('Pointcloud of Matched Features')
    ax3.legend()
    
    # 4. Bottom right: Trajectory plot
    ax4 = plt.subplot(224)
    
    # Get refined transforms from GTSAM
    refined_transforms = graph_optimization(imu_data, video_data, mini_batch_size=100)
    orientations = refined_transforms[:, :3, :3]
    positions = refined_transforms[:, :3, -1]
    
    # Plot optimized trajectory
    ax4.plot(positions[:, 0], positions[:, 2], 'b-', label='Optimized Trajectory')
    ax4.plot(positions[:, 0], positions[:, 2], 'bo', markersize=3)
    
    # Plot orientation arrows at regular intervals
    arrow_interval = max(1, len(positions) // 10)  # Show about 10 arrows
    for i in range(0, len(positions), arrow_interval):
        R = orientations[i]
        p = positions[i]
        forward = R[:, 0]  # First column of rotation matrix (x-axis)
        forward_xz = np.array([forward[0], forward[2]])
        yaw = np.arctan2(forward_xz[1], forward_xz[0]) - np.pi / 2
        ax4.arrow(p[0], p[2], 0.3 * np.cos(yaw), 0.3 * np.sin(yaw),
                 head_width=0.1, head_length=0.1, fc='r', ec='r')
    
    # Plot ground truth trajectory
    gt_points = np.array([
        [0, 0],          # origin
        [0, -1.5],       # first point
        [-4.5, -1.5],      # second point
        [-4.5, -3]         # final point
    ])
    ax4.plot(gt_points[:, 0], gt_points[:, 1], 'g--', linewidth=2, label='Ground Truth')
    ax4.plot(gt_points[:, 0], gt_points[:, 1], 'gx', markersize=8, markeredgewidth=2)
    
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Z (m)")
    ax4.set_title("Trajectory Comparison")
    ax4.axis("equal")
    ax4.grid(True)
    ax4.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('visualization_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load data
    video_data = np.load('data/video_data_store_short.npy', allow_pickle=True)
    imu_data = np.load('data/imu_data_store_short.npy', allow_pickle=True)
    
    # Use frame 250 (a bit after the middle)
    frame_idx = 250
    
    # Visualize results for the specified frame
    print(f"Visualizing frame {frame_idx} out of {len(video_data)} frames")
    visualize_results(video_data, imu_data, frame_idx=frame_idx) 