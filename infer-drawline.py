import os
import cv2
import torch
import numpy as np
from collections import deque
from tqdm import tqdm
from model.pa_lstm import PoseAttentionLSTM
from fight_module.yolo_pose_estimation import YoloPoseEstimation
from fight_module.util import calculate_angle, is_coordinate_zero

KEYPOINT_PAIRS = [
    [8, 6, 2], [11, 5, 7], [6, 8, 10], [5, 7, 9],
    [6, 12, 14], [5, 11, 13], [12, 14, 16], [11, 13, 15]
]

def extract_features(conf, xyn):
    features = []
    for a, b, c in KEYPOINT_PAIRS:
        try:
            if is_coordinate_zero(xyn[a], xyn[b], xyn[c]):
                return None
            angle = calculate_angle(xyn[a], xyn[b], xyn[c])
            avg_conf = sum([conf[a], conf[b], conf[c]]) / 3
            features.extend([angle, avg_conf])
        except IndexError:
            return None
    return features

def run_pa_lstm_inference(video_path):
    seq_len = 10
    threshold = 0.4
    lstm_model_path = "model/fight/pa_lstm_fight_model.pth"
    yolo_model_path = "model/yolo/yolov8x-pose.pt"
    output_path = f"infer-{os.path.basename(video_path)}"

    pose_estimator = YoloPoseEstimation(yolo_model_path)
    model = PoseAttentionLSTM(
        num_joints=8,
        input_dim=2,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3
    )
    model.load_state_dict(torch.load(lstm_model_path, map_location=torch.device("cpu")))
    model.eval()

    feature_buffer = deque(maxlen=seq_len)
    latest_prediction = None
    max_pair_distance = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_count = 0

    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = frame.copy()
            results = list(pose_estimator.estimate(frame))
            centers = []
            max_pair_distance = 0

            for r in results:
                if r.keypoints and r.keypoints.xy is not None and r.keypoints.conf is not None:
                    keypoints = r.keypoints.xy[0].cpu().numpy().tolist()
                    confs = r.keypoints.conf[0].cpu().numpy().tolist()
                    features = extract_features(confs, keypoints)
                    if features:
                        feature_buffer.append(features)

            if len(feature_buffer) == seq_len:
                feature_seq = np.array(feature_buffer, dtype=np.float32).reshape(seq_len, 8, 2)
                X = torch.tensor([feature_seq], dtype=torch.float32)
                with torch.no_grad():
                    logits = model(X)
                    pred = torch.sigmoid(logits)
                    latest_prediction = pred.item()

            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes.xyxy) > 0:
                    b = r.boxes.xyxy.cpu().numpy()
                    for box in b:
                        x1, y1, x2, y2 = box
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        centers.append((center_x, center_y))

            for r in results:
                try:
                    annotated_frame = r.plot(img=annotated_frame, conf=False)
                except Exception as e:
                    print(f"[WARN] Failed to plot result: {e}")

            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    pt1 = centers[i]
                    pt2 = centers[j]
                    dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
                    if dist > max_pair_distance:
                        max_pair_distance = dist
                    cv2.line(annotated_frame, pt1, pt2, (255, 255, 255), thickness=1)
                    mid_x = int((pt1[0] + pt2[0]) / 2)
                    mid_y = int((pt1[1] + pt2[1]) / 2)
                    cv2.putText(annotated_frame, f"{dist}px", (mid_x, mid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if latest_prediction is None:
                label = "WARMING UP"
                color = (200, 200, 0)
            else:
                is_fight = latest_prediction > threshold
                label_text = "Violence" if is_fight else "NORMAL"
                label = f"{label_text} ({latest_prediction:.2f})"
                color = (0, 0, 255) if is_fight else (0, 255, 0)

            cv2.putText(annotated_frame, f"{label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            if latest_prediction is not None and is_fight and max_pair_distance >= 100:
                cv2.putText(annotated_frame, "False Violence", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            out.write(annotated_frame)
            cv2.imshow("PA-LSTM Inference", annotated_frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

            frame_count += 1
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Inference complete. Saved result to: {output_path}")

if __name__ == "__main__":
    video_path = "violence-crowd.mp4"
    run_pa_lstm_inference(video_path)
