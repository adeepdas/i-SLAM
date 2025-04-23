import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_plot(R, p, frame_idx, arrow_handle):
    """
    Plots a single frame showing the current position and orientation in 2D.
    This function takes one orientation (R) and one position (p) at a time.
    It clears the old arrow (if any) but leaves the previously plotted points.
    Note: This function plots the x-z plane, not the y-z plane.
    
    Args:
        R (np.ndarray): rotation matrix of shape (3, 3)
        p (np.ndarray): position vector of shape (3,)
        frame_idx (int): Frame index.
        arrow_handle (list): A one-element list containing the current arrow object.
    """
    # Plot the current position as a blue dot.
    plt.plot(p[0], p[2], 'bo')
    
    # Remove the old arrow if it exists.
    if arrow_handle[0] is not None:
        arrow_handle[0].remove()
    
    # Extract the forward direction (body x-axis) and project onto the x-z plane
    forward = R[:, 0]  # First column of rotation matrix (x-axis)
    # Use x and z components for horizontal plane visualization
    forward_xz = np.array([forward[0], forward[2]])
    # Calculate yaw angle in the x-z plane
    yaw = np.arctan2(forward_xz[1], forward_xz[0]) - np.pi / 2
    
    # Plot a new arrow representing the orientation
    arrow_handle[0] = plt.arrow(p[0], p[2],
                                0.3 * np.cos(yaw), 0.3 * np.sin(yaw),
                                head_width=0.1, head_length=0.1, fc='r', ec='r')
    
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title(f"Frame {frame_idx}")
    plt.axis("equal")
    plt.grid(True)

def animate_trajectory(orientations, positions, interval=20):
    """
    Animate the trajectory using Matplotlib's FuncAnimation.
    
    Args:
        orientations (np.ndarray): rotation matrices of shape (N, 3, 3)
        positions (np.ndarray): position vector of shape (N, 3)
        interval (int): Delay between frames in milliseconds
        
    Returns:
        ani (FuncAnimation): The Matplotlib animation object.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.axis("equal")
    ax.grid(True)
    
    # Use a mutable container to store the arrow handle.
    arrow_handle = [None]
    
    def update(frame_idx):
        # Set the current axes so our plot() function uses ax.
        plt.sca(ax)
        # Call our plot() function which plots the current point and updates the arrow.
        create_plot(orientations[frame_idx], positions[frame_idx], frame_idx, arrow_handle)
    
    ani = animation.FuncAnimation(fig, update, frames=len(positions), interval=interval, repeat=False)
    plt.show()
    return ani

def plot_pointclouds(pc1, pc2):
    """
    Plot two pointclouds in 3D in two different colors on the same plot.

    Args:
        pc1 (np.ndarray): pointcloud of shape (N, 3)
        pc2 (np.ndarray): pointcloud of shape (N, 3)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='r', marker='o', label='Previous Frame')
    ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c='b', marker='o', label='Current Frame')
    ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.show()

def plot_trajectory(orientations, positions):
    """
    Plot a static trajectory showing positions and orientations in 2D.
    This function plots the x-z plane, not the y-z plane.
    
    Args:
        orientations (np.ndarray): rotation matrices of shape (N, 3, 3)
        positions (np.ndarray): position vectors of shape (N, 3)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Trajectory")
    ax.axis("equal")
    ax.grid(True)
    
    # Plot all positions as blue dots and line
    ax.plot(positions[:, 0], positions[:, 2], 'b-', label='Trajectory')
    ax.plot(positions[:, 0], positions[:, 2], 'bo', markersize=3)
    
    # Use a mutable container to store the arrow handle
    arrow_handle = [None]
    
    # Plot orientation arrows at regular intervals
    arrow_interval = max(1, len(positions) // 10)  # Show about 10 arrows
    for i in range(0, len(positions), arrow_interval):
        create_plot(orientations[i], positions[i], i, arrow_handle)
    
    ax.legend()
    plt.show()