import numpy as np
import cv2

# ------ Do ORB Feature Matching between two RGB numpy arrays ------

# Input: 2 RGB numpy arrays
# Output: 2 lists of good matches
#         goodMatchesP: list of good matches from imgP
#         goodMatchesQ: list of good matches from imgQ
#         goodMatchesP[i] = [x, y] of the i-th good match in imgP
#         goodMatchesQ[i] = [x, y] of the i-th good match in imgQ
# Output the visualization of the good matches as well
def feature_extraction(rgb_img_P, rgb_img_Q):
  
    # Convert images to grayscale for ORB processing
    grayP = cv2.cvtColor(rgb_img_P, cv2.COLOR_RGB2GRAY)
    grayQ = cv2.cvtColor(rgb_img_Q, cv2.COLOR_RGB2GRAY)

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=100)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(grayP, None)
    kp2, des2 = orb.detectAndCompute(grayQ, None)

    # FLANN parameters - CAN BE TUNED
    index_params = dict(algorithm = 6, # FLANN_INDEX_LSH
                        table_num = 12,
                        key_size = 12,
                        multi_probe_level = 2)
    search_params = dict(checks=100)  

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    goodMatchesP = []
    goodMatchesQ = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            goodMatchesP.append(kp1[m.queryIdx].pt)
            goodMatchesQ.append(kp2[m.trainIdx].pt)
    # print("P Img GoodMatches:\n", goodMatchesP)
    # print("Q Img GoodMatches:\n", goodMatchesQ)

    # ############################################################################################################################################################
    # # VISUALIZE THE GOOD MATCHES
    # img_combined = np.hstack((rgb_img_P, rgb_img_Q))  # Stack images horizontally

    # # Draw lines between the matching points
    # for pt1, pt2 in zip(goodMatchesP, goodMatchesQ):
    #     # pt1 is from imgP and pt2 is from imgQ
    #     # Offset the pt2 by the width of img_2d_P since img_2d_Q is to the right
    #     pt2_offset = (int(pt2[0] + rgb_img_P.shape[1]), int(pt2[1]))  # Offset pt2

    #     # Draw the line on the combined image
    #     cv2.line(img_combined, (int(pt1[0]), int(pt1[1])), pt2_offset, (0, 255, 0), 1)

    # # Show the combined image with matches
    # cv2.imshow("Good Matches", img_combined)
    # cv2.waitKey(0)  # Wait until a key is pressed
    # cv2.destroyAllWindows()
    # ############################################################################################################################################################

    return np.array(goodMatchesP, dtype=int), np.array(goodMatchesQ, dtype=int)


if __name__ == "__main__":
  
  # test if my bullshit works
  img1 = cv2.imread('test_images/test51.jpg', cv2.IMREAD_COLOR)
  img2 = cv2.imread('test_images/test52.jpg', cv2.IMREAD_COLOR)

  # Preform Feature Extraction
  img1_pts, img2_pts = feature_extraction(img1, img2)

  # Command to Run from main i-SLAM file: 
  #   python3 iSLAM/orb.py  