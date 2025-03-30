import numpy as np
from scipy.linalg import expm
from iSLAM.utils import hat, skew, transform
import matplotlib.pyplot as plt


def estimate_pose(P, Q, max_iter=10, tol=1e-6):
    """
    Estimate SE(3) transformation using Lie algebra optimization
    (Gauss-Newton with incremental accumulation)

    Args:
        P (np.ndarray): Source points of shape (N, 3)
        Q (np.ndarray): Target points of shape (N, 3)
        max_iter (int): Maximum number of iterations
        tol (float): Tolerance for convergence

    Returns:
        R (np.ndarray): Rotation matrix of shape (3, 3)
        t (np.ndarray): Translation vector of shape (3,)
    """
    assert P.shape == Q.shape
    N = P.shape[0]
    T = np.eye(4)

    for it in range(max_iter):
        A = np.zeros((6, 6))
        b = np.zeros((6,))

        P_transformed = transform(T, P)

        for i in range(N):
            p_i = P_transformed[i]
            q_i = Q[i]
            r_i = (q_i - p_i)

            # Jacobian: ∂r/∂ξ = [-I | p̂]
            J_i = np.hstack((skew(p_i), -np.eye(3)))  # 3x6

            A += J_i.T @ J_i        # Jᵢᵀ Jᵢ
            b += -J_i.T @ r_i       # -Jᵢᵀ rᵢ

        # Solve A x = b
        delta_xi = np.linalg.solve(A, b).flatten()

        # Update pose
        T_update = expm(hat(delta_xi))
        T = T_update @ T

        # Convergence check
        if np.linalg.norm(delta_xi) < tol:
            print(f"Converged at iteration {it}")
            break

    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


if __name__ == "__main__":
    # Generate a cube pointcloud (8 corners)
    cube_corners = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1]
    ])
    
    # Define cube edges (pairs of point indices that form edges)
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3),  # Front face
        (4, 5), (4, 6), (5, 7), (6, 7),  # Back face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
    
    # Create a random SE(3) transformation
    np.random.seed(42)  # For reproducibility
    
    # Random axis-angle rotation (small angle)
    angle = 0.3  # radians
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    
    # Convert to rotation matrix using exponential map
    R_true = expm(skew(angle * axis))
    
    # Small random translation
    t_true = np.random.uniform(-0.5, 0.5, 3)
    
    # Create SE(3) transformation matrix
    T_true = np.eye(4)
    T_true[:3, :3] = R_true
    T_true[:3, 3] = t_true
    
    # Apply transformation to get source points
    source_points_clean = transform(T_true, cube_corners)
    
    # Add noise to source points
    noise_level = 0.05
    noise = np.random.normal(0, noise_level, source_points_clean.shape)
    source_points = source_points_clean + noise
    
    # Estimate pose using our algorithm
    R_est, t_est = estimate_pose(source_points, cube_corners)
    
    # Create estimated transformation matrix
    T_est = np.eye(4)
    T_est[:3, :3] = R_est
    T_est[:3, 3] = t_est
    
    # Apply estimated transformation to source points
    aligned_points = transform(T_est, source_points)
    
    # Print transformation errors
    rot_error = np.linalg.norm(R_est @ R_true.T - np.eye(3), 'fro')
    trans_error = np.linalg.norm(t_est - t_true)
    print(f"Rotation error: {rot_error:.6f}")
    print(f"Translation error: {trans_error:.6f}")
    
    # Visualize the results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the target cube
    ax.scatter(cube_corners[:, 0], cube_corners[:, 1], cube_corners[:, 2], 
               c='blue', marker='o', s=100, label='Target')
    
    # Plot the aligned source cube
    ax.scatter(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2], 
               c='green', marker='x', s=100, label='Aligned Source')
    
    # Draw edges for target cube
    for edge in edges:
        p1, p2 = edge
        ax.plot([cube_corners[p1, 0], cube_corners[p2, 0]],
                [cube_corners[p1, 1], cube_corners[p2, 1]],
                [cube_corners[p1, 2], cube_corners[p2, 2]], 'b-', linewidth=2)
    
    # Draw edges for aligned source cube
    for edge in edges:
        p1, p2 = edge
        ax.plot([aligned_points[p1, 0], aligned_points[p2, 0]],
                [aligned_points[p1, 1], aligned_points[p2, 1]],
                [aligned_points[p1, 2], aligned_points[p2, 2]], 'g-', linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Cube Registration (Source → Target)')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([
        np.max([cube_corners[:, 0].max(), aligned_points[:, 0].max()]) - 
        np.min([cube_corners[:, 0].min(), aligned_points[:, 0].min()]),
        np.max([cube_corners[:, 1].max(), aligned_points[:, 1].max()]) - 
        np.min([cube_corners[:, 1].min(), aligned_points[:, 1].min()]),
        np.max([cube_corners[:, 2].max(), aligned_points[:, 2].max()]) - 
        np.min([cube_corners[:, 2].min(), aligned_points[:, 2].min()])
    ]).max() / 2.0
    
    mid_x = (cube_corners[:, 0].mean() + aligned_points[:, 0].mean()) / 2
    mid_y = (cube_corners[:, 1].mean() + aligned_points[:, 1].mean()) / 2
    mid_z = (cube_corners[:, 2].mean() + aligned_points[:, 2].mean()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()
