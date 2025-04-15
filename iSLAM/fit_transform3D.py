import numpy as np
from scipy.linalg import expm
from iSLAM.utils import hat, skew, transform, to_transform
import matplotlib.pyplot as plt


def icp(P, Q, max_iter=10, tol=1e-6):
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
            # print(f"Converged at iteration {it}")
            break

    return T

def svd_registration(P, Q):
    """
    Estimate SE(3) transformation using SVD-based method

    Args:
        P (np.ndarray): Source points of shape (N, 3)
        Q (np.ndarray): Target points of shape (N, 3)

    Returns:
        R (np.ndarray): Rotation matrix of shape (3, 3)
        t (np.ndarray): Translation vector of shape (3,)
    """
    # compute centroids
    P_centroid = np.mean(P, axis=0)
    Q_centroid = np.mean(Q, axis=0)

    # center points
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    # covariance matrix
    H = P_centered.T @ Q_centered

    # optimize
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    t = Q_centroid - R @ P_centroid

    return to_transform(R, t)

def ransac(P, Q, optim_method="svd", threshold=0.05, max_iterations=1000):    
    """
    Estimate SE(3) transformation using RANSAC

    Args:
        P (np.ndarray): Source points of shape (N, 3)
        Q (np.ndarray): Target points of shape (N, 3)
        optim_method (str): Optimization method, either "svd" or "icp"
        threshold (float): Inlier threshold
        max_iterations (int): Maximum number of iterations

    Returns:
        T (np.ndarray): SE(3) transformation matrix of shape (4, 4)
    """
    # threshold needs to be tuned to get best fit
    inliers_count_best = 0
    inliers_best = None
    
    for _ in range(max_iterations):
        # randomly select a subset of matches
        sample_indices = np.random.choice(len(P), size=3, replace=False)

        src_pts = P[sample_indices]
        target_pts = Q[sample_indices]
        
        # estimate transformation using the sampled points
        if optim_method == "svd":
            T = svd_registration(src_pts, target_pts)
        else:
            T = icp(src_pts, target_pts)
        
        # apply the transformation to all keypoints in source frame
        transformed_pts = transform(T, P)
        
        # calculate distances to keypoints in target frame
        dist = np.linalg.norm(transformed_pts - Q, axis=1)
        
        # count inliers (points with distances below the threshold)
        inliers = dist < threshold
        inliers_count = np.sum(inliers)
        
        # update the best transform if the current one has more inliers
        if inliers_count > inliers_count_best:
            inliers_count_best = inliers_count
            inliers_best = inliers
    
    # final fit
    src_pts = P[inliers_best]
    target_pts = Q[inliers_best]
    if optim_method == "svd":
        T = svd_registration(src_pts, target_pts)
    else:
        T = icp(src_pts, target_pts)
    
    return T


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
    T_true = to_transform(R_true, t_true)
    
    # Apply transformation to get source points
    source_points_clean = transform(T_true, cube_corners)
    
    # Add noise to source points
    noise_level = 0.05
    noise = np.random.normal(0, noise_level, source_points_clean.shape)
    source_points = source_points_clean + noise
    
    best_T = ransac(source_points, cube_corners, optim_method="svd")
    print("Estimated Transformation Matrix:")
    print(best_T)

    # Apply estimated transformation to source points
    aligned_points = transform(best_T, source_points)
    
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
