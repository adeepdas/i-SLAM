import numpy as np
from iSLAM.gtsam_iter import graph_optimization
from iSLAM.bow import BoWMatcher
from iSLAM.visualization import plot_trajectory, animate_trajectory

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load data
    print("Loading data...")
    imu_data = np.load('data/munger/imu_data_munger_big.npy', allow_pickle=True)
    video_data = np.load('data/munger/video_data_munger_big.npy', allow_pickle=True)
    
    # Initialize BoW matcher
    print("Initializing BoW matcher...")
    bow_matcher = BoWMatcher(vocab_size=1000)
    
    # Extract descriptors for vocabulary training (use first 10 frames)
    print("Training BoW vocabulary...")
    descriptor_list = []
    for t in range(0, len(video_data), 200):
        rgb_frame, _ = bow_matcher.rotate_frame(video_data[t])
        kp, des = bow_matcher.extract_orb_features(rgb_frame)
        if des is not None:
            descriptor_list.append(des)
    bow_matcher.train_vocab(descriptor_list)
    
    # Process frames for loop closure detection
    print("\nDetecting loop closures...")
    loop_closure_pairs = []
    for t in range(len(video_data)):
        rgb_frame, _ = bow_matcher.rotate_frame(video_data[t])
        bow_current = bow_matcher.compute_bow_descriptor(rgb_frame)
        if bow_current is None:
            continue

        # Search for loop closure among earlier frames
        loop_idx, score = bow_matcher.find_loop_candidate_kdtree(bow_current, t)
        bow_matcher.add_to_keyframe_db(bow_current, t)

        if loop_idx != -1:
            img1 = rgb_frame
            img2, _ = bow_matcher.rotate_frame(video_data[loop_idx])
            pts1, pts2, _, _ = bow_matcher.match_features(img1, img2)
            if len(pts1) > 100:
                print(f"[Loop Closure] Frame {t} ↔ Frame {loop_idx} (similarity: {score:.4f})")
                loop_closure_pairs.append((t, loop_idx))
    
    # Perform SLAM optimization
    print("\nPerforming SLAM optimization...")
    refined_transforms = graph_optimization(imu_data, video_data, mini_batch_size=100)
    
    # Visualize results
    print("\nVisualizing trajectory...")
    orientations = refined_transforms[:, :3, :3]
    positions = refined_transforms[:, :3, -1]
    
    # Plot static trajectory
    plot_trajectory(orientations, positions)
    
    # Optionally animate trajectory
    # animate_trajectory(orientations, positions, interval=1)

if __name__ == "__main__":
    main()
