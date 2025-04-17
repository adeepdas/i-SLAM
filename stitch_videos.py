import cv2
import numpy as np
import os

def stitch_videos():
    # Input video paths
    video_paths = [
        'output_videos/rgbd.mp4',
        'output_videos/matches.mp4',
        'output_videos/pointclouds.mp4',
        'output_videos/trajectory.mp4'
    ]
    
    # Check if all videos exist
    for path in video_paths:
        if not os.path.exists(path):
            print(f"Error: Video {path} not found!")
            return
    
    # Open all video files
    videos = [cv2.VideoCapture(path) for path in video_paths]
    
    # Get video properties from the first video
    width = int(videos[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(videos[0].get(cv2.CAP_PROP_FPS))
    
    # Target size for all videos (use the smallest dimensions)
    target_width = min([int(v.get(cv2.CAP_PROP_FRAME_WIDTH)) for v in videos])
    target_height = min([int(v.get(cv2.CAP_PROP_FRAME_HEIGHT)) for v in videos])
    
    print(f"Resizing all videos to {target_width}x{target_height}")
    
    # Create output video writer
    output_path = 'output_videos/combined.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width*2, target_height*2))
    
    # Read frames from all videos and combine them
    while True:
        frames = []
        for video in videos:
            ret, frame = video.read()
            if not ret:
                break
            # Resize frame to target dimensions
            frame = cv2.resize(frame, (target_width, target_height))
            frames.append(frame)
        
        # If any video ended, break
        if len(frames) != 4:
            break
        
        # Create 2x2 grid
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        combined = np.vstack((top_row, bottom_row))
        
        # Write combined frame
        out.write(combined)
    
    # Release resources
    for video in videos:
        video.release()
    out.release()
    
    print(f"Combined video saved to {output_path}")

if __name__ == "__main__":
    stitch_videos() 