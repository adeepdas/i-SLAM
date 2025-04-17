import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from iSLAM.visual_odometry import depth_to_pointcloud, rotate_frame
from iSLAM.orb import feature_extraction
from iSLAM.visualization import plot_pointclouds
from iSLAM.fit_transform3D import ransac
import os

def create_rgbd_video(frames, output_path):
    """Create a video showing RGB and depth frames side by side."""
    # Get frame dimensions
    bgr, _ = rotate_frame(frames[0])
    height, width = bgr.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width*2, height))
    
    for frame in frames:
        bgr, depth = rotate_frame(frame)
        
        # Normalize depth for visualization
        depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
        depth_8bit = depth_normalized.astype(np.uint8)
        depth_colored = cv2.applyColorMap(255-depth_8bit, cv2.COLORMAP_JET)
        
        # Combine frames
        combined = np.hstack((bgr, depth_colored))
        out.write(combined)
    
    out.release()

def create_matches_video(frames, output_path):
    """Create a video showing feature matches between consecutive frames."""
    # Get frame dimensions
    bgr, _ = rotate_frame(frames[0])
    height, width = bgr.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width*2, height))
    
    for i in range(len(frames)-1):
        bgr_prev, _ = rotate_frame(frames[i])
        bgr_curr, _ = rotate_frame(frames[i+1])
        
        # Get matches
        matches_prev, matches_curr = feature_extraction(bgr_prev, bgr_curr)
        
        # Create visualization
        combined = np.hstack((bgr_prev, bgr_curr))
        for pt1, pt2 in zip(matches_prev, matches_curr):
            pt2_offset = (int(pt2[0] + width), int(pt2[1]))
            cv2.line(combined, (int(pt1[0]), int(pt1[1])), pt2_offset, (0, 255, 0), 1)
        
        out.write(combined)
    
    out.release()

def create_pointcloud_video(frames, output_path):
    """Create a video showing pointcloud evolution."""
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scale factor for pointcloud generation
    sx = 1/6
    sy = 1/6
    
    def update(frame_idx):
        ax.clear()
        
        if frame_idx < len(frames)-1:
            frame_prev = frames[frame_idx]
            frame_curr = frames[frame_idx+1]
            
            bgr_prev, depth_prev = rotate_frame(frame_prev)
            bgr_curr, depth_curr = rotate_frame(frame_curr)
            
            # Get matches
            matches_prev, matches_curr = feature_extraction(bgr_prev, bgr_curr)
            if matches_prev.shape[0] < 100:
                return
            
            # Generate pointclouds
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
            
            # Get points corresponding to matches
            P = pc_prev[matches_prev[:, 1], matches_prev[:, 0], :]
            Q = pc_curr[matches_curr[:, 1], matches_curr[:, 0], :]
            
            # Remove points with z = 0 (filter infinite depth)
            P = P[P[:, 2] != 0]
            Q = Q[Q[:, 2] != 0]
            
            # Remove points with z < -20 (filter max depth)
            P = P[P[:, 2] > -20]
            Q = Q[Q[:, 2] > -20]
            
            # Plot pointclouds
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='r', marker='o', label='Previous Frame')
            ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c='b', marker='o', label='Current Frame')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Frame {frame_idx}')
            ax.legend()
            
            # Set equal aspect ratio
            max_range = np.array([
                np.max([P[:, 0].max(), Q[:, 0].max()]) - np.min([P[:, 0].min(), Q[:, 0].min()]),
                np.max([P[:, 1].max(), Q[:, 1].max()]) - np.min([P[:, 1].min(), Q[:, 1].min()]),
                np.max([P[:, 2].max(), Q[:, 2].max()]) - np.min([P[:, 2].min(), Q[:, 2].min()])
            ]).max() / 2.0
            
            mid_x = (P[:, 0].mean() + Q[:, 0].mean()) / 2
            mid_y = (P[:, 1].mean() + Q[:, 1].mean()) / 2
            mid_z = (P[:, 2].mean() + Q[:, 2].mean()) / 2
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(frames)-1, interval=100)
    ani.save(output_path, writer='ffmpeg', fps=10)

def create_trajectory_video(transforms, output_path):
    """Create a video showing the camera trajectory."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.axis("equal")
    ax.grid(True)
    
    positions = transforms[:, :3, 3]
    orientations = transforms[:, :3, :3]
    
    # Use a mutable container to store the arrow handle
    arrow_handle = [None]
    
    def update(frame_idx):
        # Plot all positions up to current frame
        ax.plot(positions[:frame_idx+1, 0], positions[:frame_idx+1, 2], 'b-')
        
        # Plot the current position as a blue dot
        ax.plot(positions[frame_idx, 0], positions[frame_idx, 2], 'bo')
        
        # Remove the old arrow if it exists
        if arrow_handle[0] is not None:
            arrow_handle[0].remove()
        
        # Extract the forward direction (body x-axis) and project onto the x-z plane
        forward = orientations[frame_idx, :, 0]  # First column of rotation matrix (x-axis)
        forward_xz = np.array([forward[0], forward[2]])
        yaw = np.arctan2(forward_xz[1], forward_xz[0]) - np.pi / 2
        
        # Plot a new arrow representing the orientation
        arrow_handle[0] = ax.arrow(positions[frame_idx, 0], positions[frame_idx, 2],
                                  0.3 * np.cos(yaw), 0.3 * np.sin(yaw),
                                  head_width=0.1, head_length=0.1, fc='r', ec='r')
        
        ax.set_title(f'Frame {frame_idx}')
    
    ani = FuncAnimation(fig, update, frames=len(positions), interval=100)
    ani.save(output_path, writer='ffmpeg', fps=10)

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('output_videos', exist_ok=True)
    
    # Load frames
    frames = np.load('data/rectangle_vertical.npy', allow_pickle=True)
    
    # Generate videos
    print("Generating RGB-D video...")
    # create_rgbd_video(frames, 'output_videos/rgbd.mp4')
    
    print("Generating matches video...")
    # create_matches_video(frames, 'output_videos/matches.mp4')
    
    print("Generating pointcloud video...")
    # create_pointcloud_video(frames, 'output_videos/pointclouds.mp4')
    
    # Compute transforms for trajectory video
    transforms = [np.eye(4)]
    for t in range(len(frames)-1):
        frame_prev = frames[t]
        frame_curr = frames[t+1]
        bgr_prev, depth_prev = rotate_frame(frame_prev)
        bgr_curr, depth_curr = rotate_frame(frame_curr)
        
        matches_prev, matches_curr = feature_extraction(bgr_prev, bgr_curr)
        if matches_prev.shape[0] < 100:
            continue
            
        sx = 1/6
        sy = 1/6
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
        
        P = pc_prev[matches_prev[:, 1], matches_prev[:, 0], :]
        Q = pc_curr[matches_curr[:, 1], matches_curr[:, 0], :]
        P = P[P[:, 2] != 0]
        Q = Q[Q[:, 2] != 0]
        P = P[P[:, 2] > -20]
        Q = Q[Q[:, 2] > -20]
        
        T = ransac(Q, P, threshold=0.05, max_iterations=500)
        transforms.append(transforms[-1] @ T)
    
    transforms = np.array(transforms)
    
    print("Generating trajectory video...")
    create_trajectory_video(transforms, 'output_videos/trajectory.mp4')
    
    print("All videos have been generated in the output_videos directory!") 