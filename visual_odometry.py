import iSLAM.orb as orb
import iSLAM.fit_transform3D as fit_transform3D
import iSLAM.visualization as visualization
import numpy as np
import cv2


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
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    points = np.stack((X, Y, -Z), axis=-1)

    return points

def rotate_frame(frame):
    """
    Rotate the frame by 90 degrees counterclockwise.
    """
    bgr = cv2.rotate(frame['bgr'], cv2.ROTATE_90_CLOCKWISE)
    depth = cv2.rotate(frame['depth'], cv2.ROTATE_90_CLOCKWISE)
    return bgr, depth


if __name__ == "__main__":
    # read RGBD frames from npz file
    frames = np.load('data/rectangle_vertical.npy', allow_pickle=True)
    # frames = [frames[35], frames[45]]
    T = len(frames)
    transforms = [np.eye(4)]
    for t in range(T-1):
        frame_prev = frames[t]
        frame_curr = frames[t+1]
        bgr_prev, depth_prev = rotate_frame(frame_prev)
        bgr_curr, depth_curr = rotate_frame(frame_curr)
        
        # exact BGR frames and preform feature matching betwwen frames
        matches_prev, matches_curr = orb.feature_extraction(bgr_prev, bgr_curr)
        if matches_prev.shape[0] < 3:
            continue

        # extract depth data
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
        # visualization.plot_pointclouds(pc_prev, pc_curr)
        # print(f"t: {t}, matches: {matches_prev.shape}")
        # index into pointclouds by extracted features
        # switch (x, y) to (y, x) because we are indexing into np array with opencv convention 
        P = pc_prev[matches_prev[:, 1], matches_prev[:, 0], :] # N x 3
        Q = pc_curr[matches_curr[:, 1], matches_curr[:, 0], :] # N x 3

        # optimize transform
        # TODO - tune max_iterations
        T = fit_transform3D.ransac(Q, P, max_iterations=100)
        transforms.append(transforms[-1] @ T)

    transforms = np.array(transforms)
    positions = transforms[:, :3, 3]
    orientations = transforms[:, :3, :3]
    # print(transforms)

    # visualize 
    visualization.animate_trajectory(orientations, positions)