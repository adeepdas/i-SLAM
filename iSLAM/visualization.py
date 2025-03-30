import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from iSLAM.utils import skew
import iSLAM.imu_integration as ii


def simulate_square(dt=0.01, v_const=1.0, L=2.0, T_turn=1.0):
    """
    Simulate a square trajectory using the 3D IMU integration model
    (with planar motion) and return the positions and orientations.
    
    Returns:
        positions (np.ndarray): Array of shape (N, 3) containing the 3D positions.
        orientations (list): List of 3x3 rotation matrices at each time step.
    """
    # Calculate durations for straight segments and turns.
    T_straight = L / v_const  # duration for a straight segment
    w_turn = (np.pi / 2) / T_turn  # angular rate for a 90° turn (about the z-axis)
    
    # Define the control inputs for each segment.
    segments = [
        # Segment 1: Straight
        (T_straight, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        # Segment 2: Turn 90° counterclockwise
        (T_turn, np.array([0.0, 0.0, w_turn]), 
         skew(np.array([0.0, 0.0, w_turn])) @ np.array([v_const, 0.0, 0.0])),
        # Segment 3: Straight
        (T_straight, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        # Segment 4: Turn 90°
        (T_turn, np.array([0.0, 0.0, w_turn]), 
         skew(np.array([0.0, 0.0, w_turn])) @ np.array([v_const, 0.0, 0.0])),
        # Segment 5: Straight
        (T_straight, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        # Segment 6: Turn 90°
        (T_turn, np.array([0.0, 0.0, w_turn]), 
         skew(np.array([0.0, 0.0, w_turn])) @ np.array([v_const, 0.0, 0.0])),
        # Segment 7: Straight
        (T_straight, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        # Segment 8: Turn 90° to return to original heading
        (T_turn, np.array([0.0, 0.0, w_turn]), 
         skew(np.array([0.0, 0.0, w_turn])) @ np.array([v_const, 0.0, 0.0])),
    ]
    
    # --- Initial State ---
    R = np.eye(3)
    v = R @ np.array([v_const, 0.0, 0.0])  # initial velocity along the body x-axis
    p = np.zeros(3)
    g = np.zeros(3)  # no gravity for planar motion

    positions = []
    orientations = []

    # --- Simulation Loop ---
    for (duration, omega_input, a_input) in segments:
        steps = int(duration / dt)
        # For straight segments, force the velocity to be R*[v_const, 0, 0]
        if np.linalg.norm(omega_input) < 1e-8:
            v = R @ np.array([v_const, 0.0, 0.0])
        for _ in range(steps):
            positions.append(p.copy())
            orientations.append(R.copy())
            R, v, p = ii.integrate_imu_state(R, v, p, omega_input, a_input, dt, g)
    
    positions = np.array(positions)
    return positions, orientations

def create_plot(R, p, frame_idx, arrow_handle):
    """
    Plots a single frame showing the current position and orientation in 2D.
    This function takes one orientation (R) and one position (p) at a time.
    It clears the old arrow (if any) but leaves the previously plotted points.
    
    Args:
        R (np.ndarray): rotation matrix of shape (3, 3)
        p (np.ndarray): position vector of shape (3,)
        frame_idx (int): Frame index.
        arrow_handle (list): A one-element list containing the current arrow object.
    """
    # Plot the current position as a blue dot.
    plt.plot(p[0], p[1], 'bo')
    
    # Remove the old arrow if it exists.
    if arrow_handle[0] is not None:
        arrow_handle[0].remove()
    
    # Extract the forward direction (body x-axis) and project onto the horizontal plane.
    forward = R[:, 0]
    forward_xy = forward[:2]
    yaw = np.arctan2(forward_xy[1], forward_xy[0])
    
    # Plot a new arrow representing the orientation.
    arrow_handle[0] = plt.arrow(p[0], p[1],
                                 0.3 * np.cos(yaw), 0.3 * np.sin(yaw),
                                 head_width=0.1, head_length=0.1, fc='r', ec='r')
    
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
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
    ax.set_ylabel("Y (m)")
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

if __name__ == "__main__":
    positions, orientations = simulate_square(dt=0.05)
    ani = animate_trajectory(orientations, positions, interval=20)
