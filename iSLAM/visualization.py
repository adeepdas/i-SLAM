import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from iSLAM.utils import skew
import iSLAM.imu_integration as ii


G = 9.8

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
    yaw = np.arctan2(forward_xy[1], forward_xy[0]) + np.pi / 2
    
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

def read_imu_data(file_path):
    """
    Read IMU data from a file with format:
    timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    
    Args:
        file_path (str): Path to the IMU data file
        
    Returns:
        timestamps (np.ndarray): Array of timestamps
        acc_data (np.ndarray): Array of shape (N, 3) containing accelerometer data
        gyro_data (np.ndarray): Array of shape (N, 3) containing gyroscope data
    """
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    timestamps = data[:, 0]
    acc_data = -data[:, 1:4] * G
    gyro_data = data[:, 4:7]
    return timestamps, acc_data, gyro_data

def integrate_imu_trajectory(file_path, g=np.array([0, 0, -G]), R_init=np.eye(3)):
    """
    Integrate IMU data to get trajectory.
    
    Args:
        file_path (str): Path to the IMU data file
        g (np.ndarray): Gravity vector in world frame
        
    Returns:
        positions (np.ndarray): Array of shape (N, 3) containing the 3D positions
        orientations (list): List of 3x3 rotation matrices at each time step
    """
    # Read IMU data
    timestamps, acc_data, gyro_data = read_imu_data(file_path)
    
    # Initialize state
    R = R_init  # Initial orientation
    v = np.zeros(3)  # Initial velocity
    p = np.zeros(3)  # Initial position
    
    positions = [p.copy()]
    orientations = [R.copy()]
    
    # Integration loop
    for i in range(1, len(timestamps)):
        # Calculate dt from timestamps
        dt = timestamps[i] - timestamps[i-1]
        
        # Get IMU measurements
        omega = gyro_data[i-1]  # Angular velocity
        acc = acc_data[i-1]     # Linear acceleration
        
        # Integrate state
        R, v, p = ii.integrate_imu_state(R, v, p, omega, acc, dt, g)
        
        # Store results
        positions.append(p.copy())
        orientations.append(R.copy())
    
    return np.array(positions), orientations

if __name__ == "__main__":
    # Path to IMU data file
    imu_file = "data/imu_data.csv"
    
    # Check if file exists
    if not os.path.exists(imu_file):
        print(f"Error: File {imu_file} not found. Falling back to simulated data.")
        positions, orientations = simulate_square(dt=0.05)
    else:
        # Integrate IMU data to get trajectory
        # ASSUME that identity at normal iPhone configuration
        # R_init = np.array([[0, 1, 0], 
        #                    [-1, 0, 0], 
        #                    [0, 0, 1]])
        R_init = np.eye(3)
        positions, orientations = integrate_imu_trajectory(imu_file, g=np.zeros(3), R_init=R_init)
        print(f"Integrated trajectory from {len(positions)} IMU measurements")
    
    # Animate the trajectory
    ani = animate_trajectory(orientations, positions, interval=20)
