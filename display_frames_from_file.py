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
        rgb_frame = frame['rgb']
        
        # Retrieve the raw depth data (float16) and normalize it for display
        depth_map = frame['depth']

        # Only do this if displaying depth data
        depth_map[np.isnan(depth_map)] = 0
        depth_map[np.isinf(depth_map)] = 0
        depth_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
        depth_8bit = depth_normalized.astype(np.uint8)
        colored_depth = cv2.applyColorMap(255-depth_8bit, cv2.COLORMAP_JET)


        # Display the images in separate windows.
        #rgb to bgr
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR) # cv2 uses BGR by default
        # Resize the images for better visibility
        rgb_frame = cv2.resize(rgb_frame, (640, 360))
        depth_8bit = cv2.resize(depth_8bit, (640, 360))
        colored_depth = cv2.resize(colored_depth, (640, 360))
        cv2.imshow("RGB Frame", rgb_frame)
        cv2.imshow("Depth Frame", depth_8bit)
        cv2.imshow("Colored Depth Frame", colored_depth)
        
        # Press 'q' to exit early.
        key = cv2.waitKey(TIME_BETWEEN_FRAMES_MILLISECONDS)
        if key & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'recorded_frames.npy' with the path to your .npy file.
    display_frames("recorded_frames.npy")
