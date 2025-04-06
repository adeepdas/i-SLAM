import iSLAM.orb as orb
import iSLAM.fit_transform3D as fit_transform3D
import iSLAM.visualization as visualization
import numpy as np


# MAIN
# input is 2D SQUARE array of an image
# each pixel has 4 values R, G, B, and D
# Example: at [0, 0] in the array, the value will be [R, G, B, D]
# RGB values are in the range of 0-255 and will be floating point 16 (convert to uint8)
# D values are in floating point 16 (no need to touch)

# INPUT_IMAGE_HEIGHT = 180
# INPUT_IMAGE_WIDTH = 340
# INPUT_IMAGE_CHANNELS = 4

if __name__ == "__main__":
    # read RGBD frames from npz file
    data = np.load('data/data.npz')
    frames = data['frames'] # T, H, W, D
    T, H, W, D = frames.shape
    intrinsic = data['intrinsic'] # T, 3, 3
    
    transforms = []
    for t in range(T-1):
        rgbd1 = frames[t]
        rgbd2 = frames[t+1]

        # feature extraction
        rgb1 = rgbd1[:, :, 0:3].astype(np.uint8)
        rgb2 = rgbd2[:, :, 0:3].astype(np.uint8)
        validMatches1, validMatches2 = orb.feature_extraction(rgb1, rgb2)

        # convert rgbd to point cloud
        depth1 = rgbd1[:, :, 3]
        depth2 = rgbd2[:, :, 3]
        # TODO: - Convert depth to point cloud (add z to (x, y))
        pc1 = np.zeros((H, W, 3))
        pc2 = np.zeros((H, W, 3))
        # index into pointclouds by extracted features
        P = pc1[validMatches1, :] # N x 3
        Q = pc2[validMatches2, :] # N x 3

        # optimize transform
        T = fit_transform3D.ransac(P, Q)
        transforms.append(transforms[-1] @ T)

    transforms = np.array(transforms)
    positions = transforms[:, :3, 3]
    orientations = transforms[:, :3, :3]

    # visualize 
    visualization.animate_trajectory(orientations, positions)