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
        if is_coordinate_zero(xyn[a], xyn[b], xyn[c]):
            return None
        angle = calculate_angle(xyn[a], xyn[b], xyn[c])
        avg_conf = sum([conf[a], conf[b], conf[c]]) / 3
        features.extend([angle, avg_conf])
    return features

def run_pa_lstm_inference(video_path):
    yolo_model_path = "model/yolo/yolov8x-pose.pt"
    lstm_model_path = "model/fight/pa_lstm_fight_model.pth"
    seq_len = 10
    threshold = 0.6
    output_path = f"pa_lstm_result_{os.path.basename(video_path)}"

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
            results = pose_estimator.estimate(frame)

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
                            pred = torch.sigmoid(model(X).unsqueeze(1)).item()
                        latest_prediction = pred

                    annotated_frame = r.plot()
                    break

            if len(feature_buffer) < seq_len or latest_prediction is None:
                label = "WARMING UP"
                color = (200, 200, 0)
            else:
                is_fight = latest_prediction > threshold
                label_text = "FIGHT" if is_fight else "NORMAL"
                label = f"{label_text} ({latest_prediction:.2f})"
                color = (0, 0, 255) if is_fight else (0, 255, 0)

            cv2.putText(annotated_frame, f"{label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

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
    video_path = "test_video.mp4"
    run_pa_lstm_inference(video_path)
