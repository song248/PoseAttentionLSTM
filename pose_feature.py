import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ────────────────────────────────────────────────
# 1. extract angle + confidence
# ────────────────────────────────────────────────
def angle_between(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_pose_features(keypoints, confidences):
    coordinate_for_angle = [
        [8, 6, 2], [11, 5, 7], [6, 8, 10], [5, 7, 9],
        [6, 12, 14], [5, 11, 13], [12, 14, 16], [11, 13, 15]
    ]
    features = []
    try:
        for a, b, c in coordinate_for_angle:
            angle = angle_between(keypoints[a], keypoints[b], keypoints[c])
            mean_conf = np.mean([confidences[a], confidences[b], confidences[c]])
            features.append(angle)
            features.append(mean_conf)
    except:
        return None
    return features

# ────────────────────────────────────────────────
# 2. process video folder
# ────────────────────────────────────────────────
def process_dataset(dataset_path, label, model, frame_interval=15):
    features, labels = [], []
    video_files = sorted(os.listdir(dataset_path))

    for video_file in tqdm(video_files, desc=f"[{os.path.basename(dataset_path)}]"):
        if not video_file.endswith(".mp4"):
            continue

        cap = cv2.VideoCapture(os.path.join(dataset_path, video_file))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # if frame_count % frame_interval != 0:
            #     continue

            results = model(frame, verbose=False)
            keypoints_obj = results[0].keypoints

            if keypoints_obj is None or keypoints_obj.xy is None or keypoints_obj.conf is None:
                continue

            for kp, conf in zip(keypoints_obj.xy, keypoints_obj.conf):
                kp = kp.cpu().numpy()
                conf = conf.cpu().numpy()
                feat = extract_pose_features(kp, conf)
                if feat and len(feat) == 16:
                    features.append(feat)
                    labels.append([label])
        cap.release()
    return features, labels

# ────────────────────────────────────────────────
# 3. logic
# ────────────────────────────────────────────────
def extract_all_and_save():
    yolo_model_path = "model/yolo/yolov8x-pose.pt"
    dataset_root = "dataset"
    testset_root = "testset"
    model = YOLO(yolo_model_path)

    tasks = [
        {
            "name": "TrainSet",
            "violence_path": os.path.join(dataset_root, "violence"),
            "normal_path": os.path.join(dataset_root, "normal"),
            "feature_out": "features.npy",
            "label_out": "labels.npy"
        },
        {
            "name": "TestSet",
            "violence_path": os.path.join(testset_root, "violence"),
            "normal_path": os.path.join(testset_root, "normal"),
            "feature_out": "test_features.npy",
            "label_out": "test_labels.npy"
        }
    ]

    for task in tasks:
        print(f"\n[INFO] Processing {task['name']}...")
        violence_feats, violence_labels = process_dataset(task["violence_path"], 1, model)
        normal_feats, normal_labels = process_dataset(task["normal_path"], 0, model)

        all_features = violence_feats + normal_feats
        all_labels = violence_labels + normal_labels

        print(f"[INFO] {task['name']} total samples: {len(all_features)}")
        np.save(task["feature_out"], np.array(all_features))
        np.save(task["label_out"], np.array(all_labels))
        print(f"[INFO] Saved: {task['feature_out']}, {task['label_out']}")

if __name__ == "__main__":
    extract_all_and_save()
