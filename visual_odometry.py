import iSLAM.orb as orb
import iSLAM.fit_transform3D as fit_transform3D
import iSLAM.visualization as visualization
import numpy as np

# transformation to go from IPhone coordinate frame into the visualization coordinate frame
T_PHONE2VIZ = np.array([[1, 0, 0, 0], 
                       [0, 0, 1, 0], 
                       [0, -1, 0, 0], 
                       [0, 0, 0, 1]], dtype=float)

def depth_to_pointcloud(depth, fx, fy, cx, cy):
    """
    Args:
        depth (np.ndarray): Depth image of shape (H, W)
        fx (float): Focal length in the x direction
        fy (float): Focal length in the y direction
        cx (float): Principal point in the x direction
        cy (float): Principal point in the y direction

    Returns:
        points (np.ndarray): Point cloud of shape (N, 3)
    """
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    Z = depth
    # opencv convention: use (x, y) instead of (r,c)
    Y = (uu - cx) * Z / fx
    X = (vv - cy) * Z / fy

    # negate z to match iPhone's right handed orientation
    points = np.stack((X, Y, -Z), axis=-1)
    return points

if __name__ == "__main__":
    # read RGBD frames from npz file
    frames = np.load('data/recorded_frames.npy')
    # frames = [frames[35], frames[45]]
    T = len(frames)
    transforms = [T_PHONE2VIZ]
    for t in range(T-1):
        frame_prev = frames[t]
        frame_curr = frames[t+1]
        
        # exact BGR frames and preform feature matching betwwen frames
        bgr_prev = frame_prev['bgr']
        bgr_curr = frame_curr['bgr']
        matches_prev, matches_curr = orb.feature_extraction(bgr_prev, bgr_curr)
        if matches_prev.shape[0] < 3:
            continue

        # extract depth data
        depth_prev = frame_prev['depth']
        depth_curr = frame_curr['depth']
        pc_prev = depth_to_pointcloud(
            depth_prev, 
            frame_prev['fx'],
            frame_prev['fy'],
            frame_prev['cx'],
            frame_prev['cy']
        )
        pc_curr = depth_to_pointcloud(
            depth_curr, 
            frame_curr['fx'],
            frame_curr['fy'],
            frame_curr['cx'],
            frame_curr['cy']
        )
        # print(f"t: {t}, matches: {matches_prev.shape}")
        # index into pointclouds by extracted features
        # switch (x, y) to (y, x) because we are indexing into np array with opencv convention 
        P = pc_prev[matches_prev[:, 1], matches_prev[:, 0], :] # N x 3
        Q = pc_curr[matches_curr[:, 1], matches_curr[:, 0], :] # N x 3

        # optimize transform
        # TODO - tune max_iterations
        T = fit_transform3D.ransac(P, Q, max_iterations=1000)
        transforms.append(transforms[-1] @ T)

    transforms = np.array(transforms)
    positions = transforms[:, :3, 3]
    orientations = transforms[:, :3, :3]
    # print(transforms)

    # visualize 
    visualization.animate_trajectory(orientations, positions)