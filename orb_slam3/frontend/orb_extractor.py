"""ORB feature extraction; OpenCV handles low-level FAST/BRIEF machinery."""
import cv2


class ORBExtractor:
    def __init__(self, nfeatures: int = 2000):
        self.orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8)

    def extract(self, gray):
        if gray.ndim != 2:
            raise ValueError("ORBExtractor expects a grayscale image")
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors
