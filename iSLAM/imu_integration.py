
import numpy as np
from iSLAM.utils import skew
from iSLAM.visualization import animate_trajectory


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
        R (np.ndarray): rotation matrix of shape (3, 3)
        v (np.ndarray): linear velocity of shape (3,)
        p (np.ndarray): position of shape (3,)
        omega (np.ndarray): bias-corrected angular body velocity of shape (3,)
        a (np.ndarray): bias-corrected linear body acceleration of shape (3,)
        dt (float): time step
        g (np.ndarray): gravity acceleration of shape (3,)
        
    Returns:
        (R_next, v_next, p_next) (tuple): the next state
    """
    phi = omega * dt

    Gamma0_val = Gamma0(phi)
    Gamma1_val = Gamma1(phi)
    Gamma2_val = Gamma2(phi)

    # discrete dynamic propagation
    R_next = R @ Gamma0_val
    v_next = v + (R @ Gamma1_val @ a) * dt + g * dt
    p_next = p + v * dt + (R @ Gamma2_val @ a) * (dt**2) + 0.5 * g * (dt**2)
    
    return R_next, v_next, p_next

def integrate_imu_trajectory(timestamps, gyro_data, acc_data, g=np.array([0, 0, G])):
    """
    Integrate IMU data to get trajectory of positions and orientations.
    
    Args:
        timestamps (np.ndarray): Timestamps of shape (N,)
        gyro_data (np.ndarray): Gyroscope data of shape (N, 3)
        acc_data (np.ndarray): Accelerometer data of shape (N, 3)
        g (np.ndarray): Gravity vector of shape (3,)

    Returns:
        positions (np.ndarray): Positions of shape (N, 3)
        orientations (np.ndarray): Rotation matrices of shape (N, 3, 3)
    """
    R = np.eye(3)
    v = np.zeros(3)
    p = np.zeros(3)
    
    positions = [p.copy()]
    orientations = [R.copy()]
    
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        
        omega = gyro_data[i-1]
        acc = acc_data[i-1]
        
        R, v, p = integrate_imu_state(R, v, p, omega, acc, dt, g)
        
        positions.append(p.copy())
        orientations.append(R.copy())
    
    return np.array(positions), np.array(orientations)


def read_imu_data(frames):
    """
    Read IMU data from a file with format:
    timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    
    Args:
        frames (List[dict]): List of data frames
        
    Returns:
        timestamps (np.ndarray): Timestamps of shape (N,)
        acc_data (np.ndarray): Accelerometer data of shape (N, 3)
        gyro_data (np.ndarray): Gyroscope data of shape (N, 3)
    """
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


if __name__ == "__main__":
    # read IMU data
    imu_file = "data/rectangle_vertical.npy"
    frames = np.load(imu_file, allow_pickle=True)
    timestamps, acc_data, gyro_data = read_imu_data(frames)

    # integrate IMU data to get trajectory
    # set gravity to zero because iPhone does gravity compensation
    positions, orientations = integrate_imu_trajectory(timestamps, gyro_data, acc_data, g=np.zeros(3))
    print(f"Integrated trajectory from {len(positions)} IMU measurements")

    ani = animate_trajectory(orientations, positions, interval=20)
