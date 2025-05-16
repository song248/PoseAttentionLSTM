import torch
import numpy as np
from ultralytics import YOLO
import cv2

class YoloPoseExtractor:
    def __init__(self, model_path="yolov8x-pose.pt", device=None, conf_threshold=0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)
        self.conf_threshold = conf_threshold

    def extract_keypoints(self, image):
        """
        Args:
            image: np.ndarray (H, W, 3) in BGR format
        Returns:
            List of np.ndarray: each is (J, D) for one person
        """
        results = self.model(image, conf=self.conf_threshold)[0]

        keypoints_batch = []

        for kp, conf in zip(results.keypoints.xy, results.boxes.conf):
            if conf < self.conf_threshold:
                continue
            keypoints = kp.cpu().numpy()  # shape: (J, 2)
            keypoints_batch.append(keypoints)  # append (J, 2)

        return keypoints_batch  # List of (J, 2)

    def draw_poses(self, image, keypoints_batch, radius=4, color=(0, 255, 0)):
        """
        시각화용 보조 함수 (optional)
        """
        for keypoints in keypoints_batch:
            for x, y in keypoints:
                cv2.circle(image, (int(x), int(y)), radius, color, -1)
        return image
