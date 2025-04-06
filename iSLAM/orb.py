import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import random


# ------ Do ORB Feature matching between two images ------
# ------ Visualize 2 point cloiuds as well ------

# TEST POINT CLOUDS
POINT_ClOUD_A = [[9.297, 8.081, 6.334],
                [8.715, 8.037, 1.866],
                [8.926, 5.393, 8.074],
                [8.961, 3.180, 1.101],
                [2.279, 4.271, 8.180],
                [8.607, 0.070, 5.107],
                [4.174, 2.221, 1.199],
                [3.376, 9.429, 3.232],
                [5.188, 7.030, 3.636],
                [9.718, 9.624, 2.518]]

POINT_CLOUD_B = [[ 4.275, 11.325,  0.800],
                [ 0.974,  9.945, -1.841],
                [ 7.183,  9.660,  1.071],
                [ 3.073,  6.226, -4.119],
                [ 5.578,  4.845,  5.295],
                [ 7.871,  4.799, -2.659],
                [ 2.006,  2.459, -1.052],
                [-0.724,  7.882,  3.412],
                [ 1.313,  7.429,  1.616],
                [ 0.631, 11.777, -1.451]]

# Function to visualize the point clouds
def visualize_point_clouds(pc1, pc2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='r', label='Point Cloud A')
    ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c='b', label='Point Cloud B')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Function to plot 2 images on one screen
# Used during development to visualize keypoints on images
def plot_imgs(img1, img2):
  fig = plt.figure()
  ax = fig.subplots(1, 2)
  for a in ax:
    a.set_axis_off()
  ax[0].imshow(img1)
  ax[1].imshow(img2)
  plt.show()

# use ORB to extract features
def feature_extraction(image1, image2):

    img1 = cv2.imread(image1, cv2.IMREAD_COLOR)          # queryImage
    img2 = cv2.imread(image2, cv2.IMREAD_COLOR)          # trainImage

    # Convert images to grayscale for ORB processing
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # img1 = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
    # img2 = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
    # plot_imgs(img1, img2)

    # FLANN parameters - CAN BE TUNED
    FLANN_INDEX = 6
    index_params = dict(algorithm = FLANN_INDEX, 
                        tabke_num = 12,
                        key_size = 12,
                        multi_probe_level = 1)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Lowe's Ratio Test
    goodMatches = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append(m)
    
    return goodMatches, kp1, kp2
    # # Randomly extract 3 matched keypoints and print them to the terminal
    # three_random_points = {
    #     'image_1': [],
    #     'image_2': []
    # }
    
    # if len(goodMatches) > 0:
    #     selected_matches = random.sample(goodMatches, min(3, len(goodMatches)))  # Select up to 3 matches
    #     # print("Randomly Selected 3 Keypoints):")
    #     for match in selected_matches:
    #         pt1 = kp1[match.queryIdx].pt  # Coordinates in image1
    #         three_random_points['image_1'].append(pt1)
    #         pt2 = kp2[match.trainIdx].pt  # Coordinates in image2
    #         three_random_points['image_2'].append(pt2)

    # # print(f"3 random Image1 pts: {three_random_points['image_1']}")
    # # print(f"3 random Image2 pts: {three_random_points['image_2']}")

    # # print(f"Total Number of Good Matches: {len(goodMatches)}")
    # # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, goodMatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # # Create an output image to draw matches manually
    # img_matches = cv2.hconcat([img1, img2])  # Concatenate images side by side

    # # Draw matches with thicker lines
    # for match in goodMatches:
    #     pt1 = tuple(map(int, kp1[match.queryIdx].pt))
    #     pt2 = tuple(map(int, (kp2[match.trainIdx].pt[0] + img1.shape[1], kp2[match.trainIdx].pt[1])))
    #     cv2.line(img_matches, pt1, pt2, (0, 255, 0), thickness=3)  # Adjust thickness here

    # # Display the matches
    # plt.figure(figsize=(12, 6))
    # plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    # plt.title("Feature Matches with Thicker Lines")
    # plt.show()
    # return three_random_points['image_1'], three_random_points['image_2']

if __name__ == "__main__":
    
    pc1 = np.array(POINT_ClOUD_A)
    pc2 = np.array(POINT_CLOUD_B)

    # Visualize Original Point Clouds
    # visualize_point_clouds(pc1, pc2)

    # Preform Feature Extraction
    img1_pts, img2_pts = feature_extraction('test_images/test1.jpg', 'test_images/test2.jpg')
    print(f"Image 1 Points: {img1_pts}")
    print(f"Image 2 Points: {img2_pts}") 