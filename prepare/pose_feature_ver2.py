import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

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

def match_by_center(current_boxes, ref_box):
    ref_center = ((ref_box[0] + ref_box[2]) / 2, (ref_box[1] + ref_box[3]) / 2)
    min_dist = float("inf")
    best_idx = None
    for i, box in enumerate(current_boxes):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        dist = ((ref_center[0] - cx) ** 2 + (ref_center[1] - cy) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx

def process_video(video_path, info, model):
    cap = cv2.VideoCapture(video_path)
    start_frame = info["start_frame"]
    target_index = info["target_index"]
    label = info["label"]

    keypoint_buffer = []
    ref_box = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count < start_frame:
            continue

        results = model(frame, verbose=False)
        kp_obj = results[0].keypoints
        box_obj = results[0].boxes

        if kp_obj is None or kp_obj.xy is None or kp_obj.conf is None:
            continue

        keypoints_list = kp_obj.xy
        conf_list = kp_obj.conf
        boxes = box_obj.xyxy

        if frame_count == start_frame:
            if len(keypoints_list) <= target_index:
                print(f"[WARN] target index {target_index} not found in {video_path}")
                break
            kp = keypoints_list[target_index].cpu().numpy()
            conf = conf_list[target_index].cpu().numpy()
            ref_box = boxes[target_index].cpu().numpy()
        else:
            match_idx = match_by_center(boxes, ref_box)
            if match_idx is None or match_idx >= len(keypoints_list):
                continue
            kp = keypoints_list[match_idx].cpu().numpy()
            conf = conf_list[match_idx].cpu().numpy()
            ref_box = boxes[match_idx].cpu().numpy()

        feat = extract_pose_features(kp, conf)
        if feat and len(feat) == 16:
            keypoint_buffer.append((feat, label))

    cap.release()
    return keypoint_buffer

def extract_all_and_save():
    yolo_model_path = "model/yolo/yolov8x-pose.pt"
    people_json_path = "people_index.json"
    feature_out = "features.npy"
    label_out = "labels.npy"

    with open(people_json_path, "r") as f:
        people_info = json.load(f)

    model = YOLO(yolo_model_path)
    features, labels = [], []

    for video_path, info in tqdm(people_info.items(), desc="Processing Videos"):
        if info["target_index"] is None:
            continue
        feats = process_video(video_path, info, model)
        for fvec, lbl in feats:
            features.append(fvec)
            labels.append([lbl])

    print(f"[INFO] Total samples: {len(features)}")
    np.save(feature_out, np.array(features))
    np.save(label_out, np.array(labels))
    print(f"[INFO] Saved to {feature_out}, {label_out}")

if __name__ == "__main__":
    extract_all_and_save()
