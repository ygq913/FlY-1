#!/usr/bin/env python3
"""
face_emotion_realtime_v2.py

实时情绪识别 - 使用训练好的模型
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import os

# -----------------------
# 配置
# -----------------------
MODEL_PATH = "face_landmarker.task"
EMOTION_MODEL = "emotion_model.pkl"
SCALER_FILE = "emotion_scaler.pkl"
FEATURES_FILE = "feature_names.txt"

# 情绪颜色和表情
EMOTION_CONFIG = {
    "neutral": {"color": (200, 200, 200), "emoji": "😐"},
    "happy": {"color": (0, 255, 0), "emoji": "😊"},
    "sad": {"color": (255, 100, 100), "emoji": "😢"},
    "angry": {"color": (0, 0, 255), "emoji": "😠"},
    "surprised": {"color": (0, 255, 255), "emoji": "😮"},
    "disgusted": {"color": (128, 0, 128), "emoji": "🤢"},
    "fearful": {"color": (255, 165, 0), "emoji": "😨"}
}

# -----------------------
# 加载情绪识别模型
# -----------------------
print("加载模型...")

if not os.path.exists(EMOTION_MODEL):
    print(f"❌ 找不到情绪模型: {EMOTION_MODEL}")
    print("请先运行: python train_and_test.py")
    exit(1)

emotion_model = joblib.load(EMOTION_MODEL)
scaler = joblib.load(SCALER_FILE)

# 加载特征名
with open(FEATURES_FILE, "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

print(f"✅ 模型加载成功，使用 {len(feature_names)} 个特征")

# -----------------------
# 初始化 Face Landmarker
# -----------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=True,
    min_face_detection_confidence=0.5
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# -----------------------
# 摄像头
# -----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 网格连接
mesh_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
contour_connections = mp.solutions.face_mesh.FACEMESH_CONTOURS

# 平滑
SMOOTH_ALPHA = 0.7
prev_smoothed = None
prev_blendshapes = None
prev_emotion = "neutral"
emotion_history = []

# FPS
fps = 0.0
prev_ts = time.time()

print("\n" + "=" * 50)
print("实时情绪识别")
print("=" * 50)
print("按 'q' 退出")
print("=" * 50 + "\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - prev_ts
        prev_ts = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = landmarker.detect(mp_image)
        except Exception as e:
            continue

        current_emotion = prev_emotion
        confidence = 0.0

        if result and result.face_landmarks and len(result.face_landmarks) > 0:
            landmarks = result.face_landmarks[0]
            pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

            # 平滑
            if prev_smoothed is None:
                prev_smoothed = pts.copy()
            else:
                prev_smoothed = SMOOTH_ALPHA * prev_smoothed + (1.0 - SMOOTH_ALPHA) * pts
            smooth_pts = prev_smoothed

            # BlendShapes
            blendshapes_dict = {}
            if result.face_blendshapes and len(result.face_blendshapes) > 0:
                for bs in result.face_blendshapes[0]:
                    blendshapes_dict[bs.category_name] = bs.score
                
                if prev_blendshapes is None:
                    prev_blendshapes = blendshapes_dict.copy()
                else:
                    for key in blendshapes_dict:
                        if key in prev_blendshapes:
                            prev_blendshapes[key] = SMOOTH_ALPHA * prev_blendshapes[key] + \
                                                   (1.0 - SMOOTH_ALPHA) * blendshapes_dict[key]

            # -----------------------
            # 情绪预测
            # -----------------------
            if prev_blendshapes:
                # 提取特征
                features = []
                for name in feature_names:
                    features.append(prev_blendshapes.get(name, 0.0))
                
                features = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features)
                
                # 预测
                pred = emotion_model.predict(features_scaled)[0]
                
                # 获取置信度
                if hasattr(emotion_model, 'predict_proba'):
                    proba = emotion_model.predict_proba(features_scaled)[0]
                    confidence = proba.max()
                else:
                    confidence = 1.0
                
                # 平滑预测结果（避免抖动）
                emotion_history.append(pred)
                if len(emotion_history) > 5:
                    emotion_history.pop(0)
                
                # 取最近5帧的众数
                from collections import Counter
                most_common = Counter(emotion_history).most_common(1)[0][0]
                current_emotion = most_common
                prev_emotion = current_emotion

            # -----------------------
            # 可视化
            # -----------------------
            emotion_color = EMOTION_CONFIG.get(current_emotion, {}).get("color", (255, 255, 255))
            
            # 绘制网格
            overlay = frame.copy()
            max_index = len(landmarks) - 1
            
            for (start, end) in mesh_connections:
                if start <= max_index and end <= max_index:
                    x0 = int(smooth_pts[start][0] * w)
                    y0 = int(smooth_pts[start][1] * h)
                    x1 = int(smooth_pts[end][0] * w)
                    y1 = int(smooth_pts[end][1] * h)
                    cv2.line(overlay, (x0, y0), (x1, y1), (0, 200, 0), 1)
            
            # 轮廓线（用情绪颜色）
            for (start, end) in contour_connections:
                if start <= max_index and end <= max_index:
                    x0 = int(smooth_pts[start][0] * w)
                    y0 = int(smooth_pts[start][1] * h)
                    x1 = int(smooth_pts[end][0] * w)
                    y1 = int(smooth_pts[end][1] * h)
                    cv2.line(overlay, (x0, y0), (x1, y1), emotion_color, 2)
            
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # 情绪标签（大字显示）
            emoji = EMOTION_CONFIG.get(current_emotion, {}).get("emoji", "")
            label = f"{current_emotion.upper()}"
            
            # 背景框
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            cv2.rectangle(frame, (10, 10), (30 + text_w, 60 + text_h), emotion_color, -1)
            cv2.putText(frame, label, (20, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            
            # 置信度
            cv2.putText(frame, f"Confidence: {confidence:.1%}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 关键 BlendShapes
            if prev_blendshapes:
                y_pos = 120
                key_bs = ["mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", 
                         "browInnerUp", "eyeWideLeft"]
                for bs_name in key_bs:
                    score = prev_blendshapes.get(bs_name, 0.0)
                    bar_len = int(score * 100)
                    cv2.rectangle(frame, (10, y_pos), (10 + bar_len, y_pos + 10), (0, 255, 0), -1)
                    cv2.putText(frame, f"{bs_name[:12]}: {score:.2f}", (120, y_pos + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_pos += 20

        else:
            cv2.putText(frame, "No Face Detected", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Real-time Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

print("程序退出")
