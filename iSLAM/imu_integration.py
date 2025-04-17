import numpy as np
import os
from utils import skew
from visualization import animate_trajectory


G = -9.8

def Gamma0(phi):
    """
    Computes Gamma0(phi) = exp(hat(phi)) for SO(3) using Rodrigues' formula.
    
    Args:
        phi (np.ndarray): vector of shape (3,)
        
    Returns:
        Gamma0 (np.ndarray): matrix of shape (3, 3)
    """
    theta = np.linalg.norm(phi)
    phi_hat = skew(phi)
    if theta < 1e-8:
        # Use second-order Taylor expansion for small angles:
        return np.eye(3) + phi_hat + 0.5 * (phi_hat @ phi_hat)
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / (theta**2)
        return np.eye(3) + A * phi_hat + B * (phi_hat @ phi_hat)

def Gamma1(phi):
    """
    Computes Gamma1(phi) as given in the lecture.
    
    Args:
        phi (np.ndarray): vector of shape (3,)
        
    Returns:
        Gamma1 (np.ndarray): matrix of shape (3, 3)
    """
    theta = np.linalg.norm(phi)
    phi_hat = skew(phi)
    if theta < 1e-8:
        # Taylor expansion: Gamma1 = I + 0.5*hat(phi) + 1/6*(hat(phi))^2
        return np.eye(3) + 0.5 * phi_hat + (1/6) * (phi_hat @ phi_hat)
    else:
        A = (1 - np.cos(theta)) / (theta**2)
        B = (theta - np.sin(theta)) / (theta**3)
        return np.eye(3) + A * phi_hat + B * (phi_hat @ phi_hat)

def Gamma2(phi):
    """
    Computes Gamma2(phi) as given in the lecture:
    
    Gamma2(phi) = 1/2 I + [(theta - sin(theta))/(theta^3)] hat(phi)
                + [(theta^2 + 2*cos(theta) - 2)/(theta^4)] (hat(phi))^2.
    
    Args:
        phi (np.ndarray): vector of shape (3,)
        
    Returns:
        Gamma2 (np.ndarray): matrix of shape (3, 3)
    """
    theta = np.linalg.norm(phi)
    phi_hat = skew(phi)
    if theta < 1e-8:
        # Taylor expansion: Gamma2 = 0.5*I + 1/6*hat(phi) + 1/24*(hat(phi))^2
        return 0.5 * np.eye(3) + (1/6) * phi_hat + (1/24) * (phi_hat @ phi_hat)
    else:
        A = (theta - np.sin(theta)) / (theta**3)
        B = (theta**2 + 2*np.cos(theta) - 2) / (theta**4)
        return 0.5 * np.eye(3) + A * phi_hat + B * (phi_hat @ phi_hat)

def integrate_imu_state(R, v, p, omega, a, dt, g):
    """
    Integrates the IMU state over one discrete time step using the zero-order hold
    assumption for the bias-corrected IMU measurements.
    
    The discrete dynamics are:
        R_{k+1} = R_k * Gamma0(omega*dt)
        v_{k+1} = v_k + R_k * Gamma1(omega*dt) * a * dt + g * dt
        p_{k+1} = p_k + v_k * dt + R_k * Gamma2(omega*dt) * a * dt^2 + 0.5 * g * dt^2
    
    Args:
        R (np.ndarray): rotation matrix of shape (3, 3) at time k
        v (np.ndarray): linear velocity vector of shape (3,) at time k
        p (np.ndarray): position vector of shape (3,) at time k
        omega (np.ndarray): bias-corrected angular velocity vector of shape (3,) at time k
        a (np.ndarray): bias-corrected linear acceleration vector of shape (3,) at time k.
        dt (float): time step
        g (np.ndarray): gravity vector of shape (3,)
        
    Returns:
        (R_next, v_next, p_next) (tuple): the state at time k+1
    """
    # Compute the rotation increment
    phi = omega * dt

    # Compute Gamma matrices based on phi
    Gamma0_val = Gamma0(phi)
    Gamma1_val = Gamma1(phi)
    Gamma2_val = Gamma2(phi)

    # Discrete propagation
    R_next = R @ Gamma0_val
    v_next = v + (R @ (Gamma1_val @ a)) * dt + g * dt
    p_next = p + v * dt + (R @ (Gamma2_val @ a)) * (dt**2) + 0.5 * g * (dt**2)
    
    return R_next, v_next, p_next


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
            R, v, p = integrate_imu_state(R, v, p, omega_input, a_input, dt, g)
    
    return np.array(positions), np.array(orientations)

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
    frames = np.load(file_path, allow_pickle=True)
    T = len(frames)
    timestamps = np.zeros(T)
    acc_data = np.zeros((T, 3))
    gyro_data = np.zeros((T, 3))
    for t in range(T):
        timestamps[t] = frames[t]['timestamp']
        ax, ay, az = frames[t]['ax'], frames[t]['ay'], frames[t]['az']
        gx, gy, gz = frames[t]['gx'], frames[t]['gy'], frames[t]['gz']
        acc_data[t] = np.array([ax, ay, az]) * G
        gyro_data[t] = np.array([gx, gy, gz])
    return timestamps, acc_data, gyro_data

def integrate_imu_trajectory(file_path, g=np.array([0, 0, G]), R_init=np.eye(3)):
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
        R, v, p = integrate_imu_state(R, v, p, omega, acc, dt, g)
        
        # Store results
        positions.append(p.copy())
        orientations.append(R.copy())
    
    return np.array(positions), np.array(orientations)


if __name__ == "__main__":
    # Path to IMU data file
    imu_file = "data/rectangle_vertical.npy"
    
    # Check if file exists
    if not os.path.exists(imu_file):
        print(f"Error: File {imu_file} not found. Falling back to simulated data.")
        positions, orientations = simulate_square(dt=0.05)
    else:
        # Integrate IMU data to get trajectory
        R_init = np.eye(3)
        # set gravity to zero because iPhone does gravity compensation
        positions, orientations = integrate_imu_trajectory(imu_file, g=np.zeros(3), R_init=R_init)
        print(f"Integrated trajectory from {len(positions)} IMU measurements")
    
    # Animate the trajectory
    ani = animate_trajectory(orientations, positions, interval=20)