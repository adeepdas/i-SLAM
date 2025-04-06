import iSLAM.orb as orb
import iSLAM.fit_transform3D as fit_transform3D
import numpy as np


# MAIN
# input is 2D SQUARE array of an image
# each pixel has 4 values R, G, B, and D
# Example: at [0, 0] in the array, the value will be [R, G, B, D]
# RGB values are in the range of 0-255 and will be floating point 16 (convert to uint8)
# D values are in floating point 16 (no need to touch)


INPUT_IMAGE_HEIGHT = 180
INPUT_IMAGE_WIDTH = 340
INPUT_IMAGE_DEPTH = 4



if __name__ == "__main__":
    # Example usage
    # frame1 = np.array(POINT_ClOUD_A, dtype=np.float32)
    # frame2 = np.array(POINT_CLOUD_B, dtype=np.float32)

    transform = ransac(frame1, frame2)
    print("Estimated Transformation Matrix:")
    print(transform)
