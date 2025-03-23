import numpy as np


# --- Helper Functions for 3D Integration ---

def hat(vec):
    """
    Converts a 3-vector into a 3x3 skew-symmetric matrix.
    
    Args:
        vec (np.ndarray): vector of shape (3,)
    
    Returns:
        hat (np.ndarray): matrix of shape (3, 3)
    """
    return np.array([
        [0,      -vec[2], vec[1]],
        [vec[2],  0,     -vec[0]],
        [-vec[1], vec[0],  0]
    ])

def Gamma0(phi):
    """
    Computes Gamma0(phi) = exp(hat(phi)) for SO(3) using Rodrigues' formula.
    
    Args:
        phi (np.ndarray): vector of shape (3,)
        
    Returns:
        Gamma0 (np.ndarray): matrix of shape (3, 3)
    """
    theta = np.linalg.norm(phi)
    phi_hat = hat(phi)
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
    phi_hat = hat(phi)
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
    phi_hat = hat(phi)
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