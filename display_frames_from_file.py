import numpy as np
import cv2

TIME_BETWEEN_FRAMES_MILLISECONDS = 200

def display_frames(npy_filename):
    # Load the recorded frames from the .npy file.
    frames = np.load(npy_filename)
    
    # Loop over each frame
    for idx, frame in enumerate(frames):
        # Prepare metadata info as text (for printing)
        info_text = (
            f"Frame {idx+1}:\n"
            f"  Timestamp: {frame['timestamp']}\n"
            f"  Intrinsics: fx={frame['fx']}, fy={frame['fy']}, cx={frame['cx']}, cy={frame['cy']}\n"
            f"  IMU Acc: ax={frame['ax']}, ay={frame['ay']}, az={frame['az']}\n"
            f"  IMU Gyro: gx={frame['gx']}, gy={frame['gy']}, gz={frame['gz']}\n"
        )
        print(info_text)
        
        # Retrieve the RGB image (already uint8 with shape (180,320,3))
        bgr_frame = frame['bgr']
        
        # Retrieve the raw depth data (float16) and normalize it for display
        depth_map = frame['depth']

        # Only do this if displaying depth data
        depth_map[np.isnan(depth_map)] = 0
        depth_map[np.isinf(depth_map)] = 0
        depth_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
        depth_8bit = depth_normalized.astype(np.uint8)
        colored_depth = cv2.applyColorMap(255-depth_8bit, cv2.COLORMAP_JET)


        # Display the images in separate windows.

        cv2.imshow("Video Frame", bgr_frame)
        cv2.imshow("Depth Frame", depth_8bit)
        cv2.imshow("Colored Depth Frame", colored_depth)
        
        # Press 'q' to exit early.
        key = cv2.waitKey(TIME_BETWEEN_FRAMES_MILLISECONDS)
        if key & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_frames("recorded_frames.npy")
