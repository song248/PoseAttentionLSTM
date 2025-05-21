import os
import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO

def draw_bbox_and_indices(image, boxes_xyxy, keypoints_xy):
    for idx, (box, kp) in enumerate(zip(boxes_xyxy, keypoints_xy)):
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 텍스트 위치 (bbox 위쪽)
        text = str(idx)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20

        cv2.putText(
            image,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            lineType=cv2.LINE_AA
        )
    return image

def find_first_frame_with_people(video_path, model, max_frames=150):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        results = model(frame, verbose=False)
        if results[0].keypoints and results[0].keypoints.xy is not None:
            if len(results[0].keypoints.xy) > 0:
                cap.release()
                return frame, results[0].keypoints.xy, results[0].boxes.xyxy, frame_idx
    cap.release()
    return None, None, None, None

def process_folder(folder_path, label, class_name, model, label_images_dir, attacker_info, missed_videos):
    save_dir = os.path.join(label_images_dir, class_name)
    os.makedirs(save_dir, exist_ok=True)
    
    video_exts = [".mp4", ".avi", ".mov", ".mkv"]

    for video_name in sorted(os.listdir(folder_path)):
        if not any(video_name.lower().endswith(ext) for ext in video_exts):
            continue
        video_path = os.path.join(folder_path, video_name)
        full_key = os.path.join(folder_path, video_name).replace("\\", "/")

        print(f"[INFO] Processing: {full_key}")
        frame, keypoints_xy, boxes_xyxy, frame_idx = find_first_frame_with_people(video_path, model)

        if frame is None:
            print(f"[WARN] No person detected in {video_name}")
            missed_videos[class_name].append(video_name)
            continue

        annotated = draw_bbox_and_indices(frame.copy(), boxes_xyxy, keypoints_xy)
        if "mp4" in video_name:
            save_name = video_name.replace(".mp4", ".jpg")
        elif "avi" in video_name:
            save_name = video_name.replace(".avi", ".jpg")
        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path, annotated)

        num_people = len(keypoints_xy)
        attacker_info[full_key] = {
            "start_frame": frame_idx,
            "target_index": None,
            "target_candidates": list(range(num_people)),
            "label": label
        }

def main():
    model_path = "model/yolo/yolov8x-pose.pt"
    dataset_root = "dataset"
    label_output = "attackers.json"
    label_images_dir = "label_frames"

    os.makedirs(label_images_dir, exist_ok=True)
    model = YOLO(model_path)
    attacker_info = {}
    missed_videos = {"violence": [], "normal": []}

    process_folder(
        folder_path=os.path.join(dataset_root, "violence"),
        label=1,
        class_name="violence",
        model=model,
        label_images_dir=label_images_dir,
        attacker_info=attacker_info,
        missed_videos=missed_videos
    )

    process_folder(
        folder_path=os.path.join(dataset_root, "normal"),
        label=0,
        class_name="normal",
        model=model,
        label_images_dir=label_images_dir,
        attacker_info=attacker_info,
        missed_videos=missed_videos
    )

    with open(label_output, "w") as f:
        json.dump(attacker_info, f, indent=4)

    for cls in ["violence", "normal"]:
        if missed_videos[cls]:
            miss_log = f"missed_{cls}.txt"
            with open(miss_log, "w") as f:
                for vid in missed_videos[cls]:
                    f.write(vid + "\n")
            print(f"⚠️ Missed {len(missed_videos[cls])} {cls} videos (no person detected). See: {miss_log}")

    print(f"\n✅ Labeling template generated: {label_output}")
    print(f"🖼️  Labeled frames saved in: {label_images_dir}/[violence|normal]/")

if __name__ == "__main__":
    main()
