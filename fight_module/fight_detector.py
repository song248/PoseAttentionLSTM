import torch
import torch.nn as nn
from fight_module.util import *


class ThreeLayerClassifier(nn.Module):
    """
    Neural network model with three layers for classification.
    """

    def __init__(self, input_size, hidden_size1, output_size):
        super(ThreeLayerClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size1, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class FightDetector:
    """
    Fight detection module using a deep learning model.
    """

    def __init__(self, fight_model, threshold=0.5, conclusion_threshold=2, final_threshold=15):
        # Load pre-trained model
        self.input_size = 16
        self.hidden_size = 8
        self.output_size = 1
        self.model = ThreeLayerClassifier(self.input_size, self.hidden_size, self.output_size)
        self.model.load_state_dict(torch.load(fight_model))
        self.model.eval()  # Set to evaluation mode

        # Define keypoints for calculating angles
        self.coordinate_for_angel = [
            [8, 6, 2],      # right shoulder - right elbow - right wrist
            [11, 5, 7],     # left shoulder - left elbow - left wrist
            [6, 8, 10],     # right elbow - right wrist - right hand
            [5, 7, 9],      # left elbow - left wrist - left hand
            [6, 12, 14],    # right elbow - right hip - right knee
            [5, 11, 13],    # left elbow - left hip - left knee
            [12, 14, 16],   # right hip - right knee - right ankle
            [11, 13, 15]    # left hip - left knee - left ankle
        ]

        # Set detection thresholds (from arguments)
        self.threshold = threshold
        self.conclusion_threshold = conclusion_threshold
        self.final_threshold = final_threshold

        # Event state variable
        self.fight_detected = 0

    def detect(self, conf, xyn):
        """
        Detects fight action based on keypoints and confidence scores.

        Args:
            conf (list): Confidence scores for each keypoint.
            xyn (list): Coordinates (x, y, visibility) for each keypoint.

        Returns:
            bool: True if fight detected, False otherwise.
        """
        input_list = []
        keypoint_unseen = False

        for n in self.coordinate_for_angel:
            first, mid, end = n[0], n[1], n[2]
            c1, c2, c3 = xyn[first], xyn[mid], xyn[end]

            if is_coordinate_zero(c1, c2, c3):
                keypoint_unseen = True
                break
            else:
                input_list.append(calculate_angle(c1, c2, c3))
                conf1, conf2, conf3 = conf[first], conf[mid], conf[end]
                input_list.append(torch.mean(torch.Tensor([conf1, conf2, conf3])).item())

        if keypoint_unseen:
            return False

        prediction = self.model(torch.Tensor(input_list))

        if prediction.item() > self.threshold:
            self.fight_detected += 1
        else:
            if self.fight_detected > 0:
                self.fight_detected -= self.conclusion_threshold

        return self.fight_detected > self.final_threshold
