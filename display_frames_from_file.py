import numpy as np
import cv2

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

        depth_map[np.isnan(depth_map)] = 0
        depth_map[np.isinf(depth_map)] = 0

        depth_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
        depth_8bit = depth_normalized.astype(np.uint8)

        # # get rid of nan and inf values
        # depth_frame = np.nan_to_num(depth_frame, nan=0.0, posinf=0.0, neginf=0.0)

        # depth_disp = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        # depth_disp = depth_disp.astype(np.uint8)
        
        # Display the images in separate windows.
        cv2.imshow("RGB Frame", rgb_frame)
        cv2.imshow("Depth Frame", depth_8bit)
        
        # Wait for 1 second (1000 ms). Press 'q' to exit early.
        key = cv2.waitKey(1000)
        if key & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'recorded_frames.npy' with the path to your .npy file.
    display_frames("recorded_frames.npy")
