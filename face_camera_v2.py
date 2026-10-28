#!/usr/bin/env python3
"""
face_camera_v2.py

增强版数据采集 - 使用自定义特征
显示更准确的表情指标
"""

import time
import csv
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from custom_features import extract_custom_features, get_feature_names

# -----------------------
# 配置
# -----------------------
MODEL_PATH = "face_landmarker.task"
OUTPUT_CSV = "face_custom_features.csv"
NUM_FACES = 1
SMOOTH_ALPHA = 0.7

# -----------------------
# 初始化
# -----------------------
print("初始化 Face Landmarker...")
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=NUM_FACES,
    output_face_blendshapes=True,
    min_face_detection_confidence=0.5
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# CSV 初始化
feature_names = get_feature_names()
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        header = feature_names + ['timestamp_ms']
        writer.writerow(header)
    print(f"✅ 创建 CSV: {OUTPUT_CSV}")

# 摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 网格
mesh_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
contour_connections = mp.solutions.face_mesh.FACEMESH_CONTOURS

# 状态
prev_smoothed = None
prev_features = None
start_time = time.time()
frame_count = 0
fps = 0.0
prev_ts = time.time()

print("\n" + "=" * 50)
print("增强版数据采集 - 自定义特征")
print("=" * 50)
print("按 'q' 退出, 按 's' 截图")
print("=" * 50 + "\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()
        timestamp_ms = int((now - start_time) * 1000)

        # FPS
        dt = now - prev_ts
        prev_ts = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = landmarker.detect(mp_image)
        except:
            continue

        if result and result.face_landmarks and len(result.face_landmarks) > 0:
            landmarks = result.face_landmarks[0]
            pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

            # 平滑
            if prev_smoothed is None:
                prev_smoothed = pts.copy()
            else:
                prev_smoothed = SMOOTH_ALPHA * prev_smoothed + (1.0 - SMOOTH_ALPHA) * pts
            smooth_pts = prev_smoothed

            # 提取自定义特征
            features = extract_custom_features(smooth_pts)
            
            # 平滑特征
            if prev_features is None:
                prev_features = features.copy()
            else:
                for key in features:
                    if key in prev_features:
                        prev_features[key] = SMOOTH_ALPHA * prev_features[key] + \
                                            (1.0 - SMOOTH_ALPHA) * features[key]

            # 保存 CSV
            row = [prev_features.get(name, 0.0) for name in feature_names]
            row.append(timestamp_ms)
            with open(OUTPUT_CSV, "a", newline="") as f:
                csv.writer(f).writerow(row)

            # -----------------------
            # 可视化
            # -----------------------
            overlay = frame.copy()
            max_index = len(landmarks) - 1

            # 网格
            for (start, end) in mesh_connections:
                if start <= max_index and end <= max_index:
                    x0 = int(smooth_pts[start][0] * w)
                    y0 = int(smooth_pts[start][1] * h)
                    x1 = int(smooth_pts[end][0] * w)
                    y1 = int(smooth_pts[end][1] * h)
                    cv2.line(overlay, (x0, y0), (x1, y1), (0, 200, 0), 1)

            # 轮廓
            for (start, end) in contour_connections:
                if start <= max_index and end <= max_index:
                    x0 = int(smooth_pts[start][0] * w)
                    y0 = int(smooth_pts[start][1] * h)
                    x1 = int(smooth_pts[end][0] * w)
                    y1 = int(smooth_pts[end][1] * h)
                    cv2.line(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)

            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # -----------------------
            # 显示关键特征（右侧面板）
            # -----------------------
            panel_w = 350
            panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
            
            y_pos = 30
            cv2.putText(panel, "Custom Features", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
            
            # 关键特征显示
            key_features = [
                ("Mouth Corner Lift", "mouth_corner_lift_avg", (-0.05, 0.05)),
                ("Mouth Open", "mouth_aspect_ratio", (0, 0.5)),
                ("Mouth Pucker", "mouth_pucker", (0, 1)),
                ("Frown Intensity", "frown_intensity", (0, 1)),
                ("Brow Height", "brow_height_avg", (0, 0.1)),
                ("Brow Inner Dist", "brow_inner_distance", (0.1, 0.2)),
                ("Eye Openness", "eye_openness_avg", (0.1, 0.5)),
                ("Smile Index", "smile_index", (-0.1, 0.1)),
                ("Sad Index", "sad_index", (0, 0.1)),
                ("Angry Index", "angry_index", (0, 1)),
                ("Surprise Index", "surprise_index", (0, 1)),
            ]
            
            for label, key, (min_val, max_val) in key_features:
                value = prev_features.get(key, 0.0)
                
                # 归一化到 0-1
                norm_val = (value - min_val) / (max_val - min_val + 1e-8)
                norm_val = max(0, min(1, norm_val))
                
                # 条形图
                bar_len = int(norm_val * 200)
                
                # 颜色：绿色(低) -> 黄色(中) -> 红色(高)
                if norm_val < 0.5:
                    color = (0, int(255 * norm_val * 2), 255)
                else:
                    color = (0, 255, int(255 * (1 - norm_val) * 2))
                
                cv2.rectangle(panel, (10, y_pos), (10 + bar_len, y_pos + 12), color, -1)
                cv2.rectangle(panel, (10, y_pos), (210, y_pos + 12), (100, 100, 100), 1)
                
                # 文字
                text = f"{label}: {value:.3f}"
                cv2.putText(panel, text, (10, y_pos + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                y_pos += 25
            
            # 表情判断
            y_pos += 20
            cv2.putText(panel, "Expression Analysis", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_pos += 25
            
            # 简单的表情判断逻辑
            smile = prev_features.get('smile_index', 0)
            frown = prev_features.get('frown_intensity', 0)
            surprise = prev_features.get('surprise_index', 0)
            mouth_open = prev_features.get('mouth_aspect_ratio', 0)
            
            expression = "Neutral"
            if smile > 0.02:
                expression = "Happy 😊"
            elif frown > 0.3:
                expression = "Angry 😠"
            elif surprise > 0.4 and mouth_open > 0.2:
                expression = "Surprised 😮"
            elif prev_features.get('mouth_corner_lift_avg', 0) < -0.01:
                expression = "Sad 😢"
            
            cv2.putText(panel, f"Detected: {expression}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 合并
            frame = np.hstack([frame, panel])

        else:
            cv2.putText(frame, "No Face Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Camera V2 - Custom Features", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"📸 截图: {filename}")

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

print(f"\n数据已保存到: {OUTPUT_CSV}")
print(f"共 {frame_count} 帧")
