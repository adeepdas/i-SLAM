import numpy as np
import cv2
import os
import pickle
from visual_odometry import rotate_frame

class BoWMatcher:
    def __init__(self, vocab_path="vocab.pkl", vocab_size=1000):
        self.vocab_path = vocab_path
        self.vocab_size = vocab_size
        self.orb = cv2.ORB_create(nfeatures=1000)
        # Use Hamming-based BFMatcher for binary descriptors
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.bow_extractor = None
        self._load_or_train_vocab = False

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


if __name__ == "__main__":
    np.random.seed(42)

    # Load frame paths from npy file
    frames = np.load('data/old_unsync_data/rectangle_vertical.npy', allow_pickle=True)

    # Initialize matcher
    matcher = BoWMatcher(vocab_size=1000)

    # Extract descriptors for vocabulary training (use first 10 frames)
    descriptor_list = []
    for t in range(10):
        #print("loop 1 - frame:", t)
        rgb_frame, _ = rotate_frame(frames[t])
        kp, des = matcher.extract_orb_features(rgb_frame)
        if des is not None:
            # descriptor_list.append(des.astype(np.float32))
            descriptor_list.append(des)
    matcher.train_vocab(descriptor_list)

    # Process all pairs of frames (t and t+1)
    for t in range(20, len(frames) - 1):
        print("loop 2 - frame:", t)
        img1, _ = rotate_frame(frames[t])
        img2, _ = rotate_frame(frames[t - 20])

        # Feature point matching
        pts1, pts2, _, _ = matcher.match_features(img1, img2)
        print(f"[Frame {t}] Matched {len(pts1)} features.")

        # BoW vector comparison
        bow1 = matcher.compute_bow_descriptor(img1)
        bow2 = matcher.compute_bow_descriptor(img2)
        if bow1 is not None and bow2 is not None:
            similarity = matcher.compare_bow(bow1, bow2)
            print(f"[Frame {t}] BoW cosine similarity: {similarity:.4f}")
        else:
            print(f"[Frame {t}] Could not compute BoW descriptor.")
