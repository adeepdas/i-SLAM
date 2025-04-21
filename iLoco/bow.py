import numpy as np
import cv2
import os
import pickle
from iLoco.visual_odometry import rotate_frame
from scipy.spatial import KDTree  


def draw_matches(img1, img2, pts1, pts2, window_name="Matches"):
    """
    Stitches img1|img2 side by side and draws green circles + lines
    between each corresponding point in pts1 ↔ pts2.
    pts1, pts2 should be arrays of shape (N,2), dtype=int.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # make blank canvas
    canvas = np.zeros((max(h1,h2), w1+w2, 3), dtype=img1.dtype)
    canvas[:h1, :w1]         = img1
    canvas[:h2, w1:w1+w2]    = img2

    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        # draw keypoints
        cv2.circle(canvas, (x1, y1),  3, (0,255,0), -1)
        cv2.circle(canvas, (x2+w1, y2), 3, (0,255,0), -1)
        # connect them
        cv2.line(canvas, (x1, y1), (x2+w1, y2), (0,255,0), 1)

    cv2.imshow(window_name, canvas)
    cv2.waitKey(500)
    
class BoWMatcher:
    def __init__(self, vocab_path="vocab.pkl", vocab_size=1000):
        self.vocab_path = vocab_path
        self.vocab_size = vocab_size
        self.orb = cv2.ORB_create(nfeatures=1000)
        # Use Hamming-based BFMatcher for binary descriptors
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.bow_extractor = None
        self._load_or_train_vocab = False
        self.keyframe_db = []           # (frame_index, bow_descriptor)
        self.bow_matrix = []            # flattened BoW vectors
        self.kdtree = None              # KDTree for fast matching

    def extract_orb_features(self, rgb_img):
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def train_vocab(self, descriptor_list):
        print("[INFO] Training visual vocabulary...")
        bow_trainer = cv2.BOWKMeansTrainer(self.vocab_size)
        for des in descriptor_list:
            if des is not None:
                # cluster expects float32
                bow_trainer.add(des.astype(np.float32))
        # 1) Cluster → CV_32F vocabulary
        vocabulary = bow_trainer.cluster()

        # ── Fix #2: quantize to uint8 so it matches ORB’s CV_8U descriptors ──
        vocab_uint8 = np.uint8(np.round(vocabulary))
        # ────────────────────────────────────────────────────────────────

        # 2) Save quantized vocab
        with open(self.vocab_path, 'wb') as f:
            pickle.dump(vocab_uint8, f)

        # 3) Create extractor with uint8 vocab
        self._create_bow_extractor(vocab_uint8)
        self._load_or_train_vocab = True


    def load_vocab(self):
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError("Vocabulary file not found.")
        with open(self.vocab_path, 'rb') as f:
            vocabulary = pickle.load(f)
        self._create_bow_extractor(vocabulary)
        self._load_or_train_vocab = True

    def _create_bow_extractor(self, vocabulary):
        # Now uses BFMatcher instead of FLANN
        self.bow_extractor = cv2.BOWImgDescriptorExtractor(self.orb, self.matcher)
        self.bow_extractor.setVocabulary(vocabulary)

    def compute_bow_descriptor(self, rgb_img):
        if not self._load_or_train_vocab:
            raise RuntimeError("Vocabulary not loaded or trained.")
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        keypoints = self.orb.detect(gray, None)
        return self.bow_extractor.compute(gray, keypoints)

    def match_features(self, img1, img2):
        kp1, des1 = self.extract_orb_features(img1)
        kp2, des2 = self.extract_orb_features(img2)

        if des1 is None or des2 is None:
            return [], [], des1, des2

        flann = cv2.FlannBasedMatcher(dict(algorithm=6, table_number=12, key_size=12, multi_probe_level=2), dict(checks=100))
        matches = flann.knnMatch(des1, des2, k=2)

        goodP, goodQ = [], []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                goodP.append(kp1[m.queryIdx].pt)
                goodQ.append(kp2[m.trainIdx].pt)

        return np.round(goodP, 0).astype(int), np.round(goodQ, 0).astype(int), des1, des2

    def compare_bow(self, bow1, bow2):
        return float(np.dot(bow1, bow2.T) / (np.linalg.norm(bow1) * np.linalg.norm(bow2)))
    
    def add_to_keyframe_db(self, bow_descriptor, frame_index):
        """
        Adds the current frame's BoW descriptor to the keyframe database and rebuilds the KDTree.
        """
        self.keyframe_db.append((frame_index, bow_descriptor))
        self.bow_matrix.append(bow_descriptor.flatten())
        
        # Rebuild KDTree from updated BoW matrix
        self.kdtree = KDTree(np.array(self.bow_matrix))
    
    def find_loop_candidate_kdtree(self, bow_current, current_index, min_gap=150, sim_thresh=0.85, top_k=10):
        """
        Searches for loop closure candidates using KDTree and cosine similarity.
        
        Parameters:
            bow_current  – BoW descriptor for current frame
            current_index – Frame index of current frame
            min_gap      – Skip frames too close to current (to avoid temporal neighbors)
            sim_thresh   – Minimum cosine similarity to consider a loop closure
            top_k        – Number of top matches from KDTree to evaluate

        Returns:
            best_match (int): Frame index of best loop closure match (or -1 if none)
            best_score (float): Cosine similarity score
        """
        if self.kdtree is None or len(self.bow_matrix) < 2:
            return -1, 0.0

        bow_flat = bow_current.flatten()
        dists, idxs = self.kdtree.query(bow_flat, k=top_k)

        best_match = -1
        best_score = 0.0

        for idx in idxs:            
            # Ensure idx is within bounds
            if idx >= len(self.keyframe_db):
                continue
            
            frame_idx, bow_vec = self.keyframe_db[idx]
        
            # Avoid matches that are too recent
            if abs(current_index - frame_idx) < min_gap:
                continue

            sim = self.compare_bow(bow_current, bow_vec)
            if sim > best_score and sim > sim_thresh:
                best_score = sim
                best_match = frame_idx

        return best_match, best_score



if __name__ == "__main__":
    np.random.seed(42)

    # Load frame paths from npy file
    frames = np.load('video_data_munger_big.npy', allow_pickle=True)

    # Initialize matcher
    matcher = BoWMatcher(vocab_size=1000)

    # Extract descriptors for vocabulary training (use first 10 frames)
    descriptor_list = []
    for t in range (0, len(frames), 200):
        #print("loop 1 - frame:", t)
        rgb_frame = rotate_frame(frames[t])
        kp, des = matcher.extract_orb_features(rgb_frame)
        if des is not None:
            # descriptor_list.append(des.astype(np.float32))
            descriptor_list.append(des)
    matcher.train_vocab(descriptor_list)

    # Process all pairs of frames (t and t+1)
    print("\n=== Realistic Loop Closure Simulation ===")
    for t in range(len(frames)):
        rgb_frame = rotate_frame(frames[t])
        bow_current = matcher.compute_bow_descriptor(rgb_frame)
        if bow_current is None:
            continue

        # Search for loop closure among earlier frames
        loop_idx, score = matcher.find_loop_candidate_kdtree(bow_current, t)
        matcher.add_to_keyframe_db(bow_current, t)

        if loop_idx != -1:
            img1 = rgb_frame
            img2 = rotate_frame(frames[loop_idx])
            pts1, pts2, _, _ = matcher.match_features(img1, img2)
            if len(pts1) > 100:
                print(f"[Loop Closure] Frame {t} ↔ Frame {loop_idx} (similarity: {score:.4f})")
                draw_matches(img1, img2, pts1, pts2)            
                print(f"[Frame {t}] BoW cosine similarity: {score:.4f}")