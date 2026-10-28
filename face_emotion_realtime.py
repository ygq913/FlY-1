#!/usr/bin/env python3
"""
face_emotion_realtime.py

实时情绪识别 - 集成版
结合 Face Landmarker + 训练好的情绪识别模型
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
EMOTION_MODEL_PATH = "emotion_model.pkl"
SCALER_PATH = "emotion_scaler.pkl"
NUM_FACES = 1
SMOOTH_ALPHA = 0.7
FPS_SMOOTH = 0.9

# 情绪颜色映射
EMOTION_COLORS = {
    "neutral": (200, 200, 200),
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "surprised": (255, 255, 0),
    "disgusted": (128, 0, 128),
    "fearful": (255, 165, 0)
}

# -----------------------
# 加载情绪识别模型
# -----------------------
emotion_model = None
scaler = None

if os.path.exists(EMOTION_MODEL_PATH) and os.path.exists(SCALER_PATH):
    emotion_model = joblib.load(EMOTION_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("情绪识别模型已加载")
else:
    print("警告：未找到情绪识别模型，将只显示 BlendShapes")
    print("请先运行: python train_emotion_model.py label")
    print("         python train_emotion_model.py train")

# -----------------------
# 特征提取函数（与训练时保持一致）
# -----------------------
def extract_features_from_landmarks(landmarks, w, h):
    """从 landmarks 提取几何特征"""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # 嘴部
    mouth_left = pts[61]
    mouth_right = pts[291]
    mouth_top = pts[13]
    mouth_bottom = pts[14]
    
    mouth_width = np.linalg.norm(mouth_left - mouth_right)
    mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
    mouth_aspect_ratio = mouth_height / (mouth_width + 1e-6)
    
    # 眼睛
    left_eye_top = pts[159]
    left_eye_bottom = pts[145]
    right_eye_top = pts[386]
    right_eye_bottom = pts[374]
    
    left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
    right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
    eye_avg_height = (left_eye_height + right_eye_height) / 2
    
    # 眉毛
    left_brow = pts[55]
    right_brow = pts[285]
    brow_distance = np.linalg.norm(left_brow - right_brow)
    
    # 鼻子
    nose_tip = pts[1]
    nose_bridge = pts[6]
    nose_length = np.linalg.norm(nose_tip - nose_bridge)
    
    # 脸部宽度
    left_face = pts[234]
    right_face = pts[454]
    face_width = np.linalg.norm(left_face - right_face)
    
    return np.array([
        mouth_width, mouth_height, mouth_aspect_ratio,
        left_eye_height, right_eye_height, eye_avg_height,
        brow_distance, nose_length, face_width
    ])

def extract_features_from_blendshapes(blendshapes_dict):
    """从 BlendShapes 提取特征"""
    # 按固定顺序提取
    feature_names = [
        "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
        "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
        "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
        "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
        "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
        "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
        "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
        "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
        "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
        "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
        "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
        "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
        "noseSneerRight"
    ]
    
    features = []
    for name in feature_names:
        features.append(blendshapes_dict.get(name, 0.0))
    
    return np.array(features)

# -----------------------
# 初始化 Face Landmarker
# -----------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

def _dummy_callback(result, output_image, timestamp_ms):
    return

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=NUM_FACES,
    output_face_blendshapes=True,
    result_callback=_dummy_callback
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# -----------------------
# 摄像头
# -----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

# 网格连接
mesh_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
contour_connections = mp.solutions.face_mesh.FACEMESH_CONTOURS

# 平滑
prev_smoothed = None
prev_blendshapes = None
prev_emotion = "neutral"

# FPS
fps = 0.0
prev_ts = time.time()
start_time = time.time()

print("实时情绪识别启动")
print("按 'q' 退出")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        timestamp_ms = int((now - start_time) * 1000)

        # FPS
        dt = now - prev_ts
        prev_ts = now
        if dt > 0:
            inst_fps = 1.0 / dt
            fps = FPS_SMOOTH * fps + (1.0 - FPS_SMOOTH) * inst_fps if fps else inst_fps

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_async(mp_image, timestamp_ms)

        current_emotion = prev_emotion
        emotion_confidence = 0.0

        if result and result.face_landmarks:
            landmarks = result.face_landmarks[0]
            
            # 转换为 numpy
            pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

            # 平滑
            if prev_smoothed is None:
                prev_smoothed = pts.copy()
            else:
                prev_smoothed = SMOOTH_ALPHA * prev_smoothed + (1.0 - SMOOTH_ALPHA) * pts

            smooth_pts = prev_smoothed

            # BlendShapes
            blendshapes_dict = {}
            if result.face_blendshapes:
                blendshapes = result.face_blendshapes[0]
                for bs in blendshapes:
                    blendshapes_dict[bs.category_name] = bs.score
                
                # 平滑
                if prev_blendshapes is None:
                    prev_blendshapes = blendshapes_dict.copy()
                else:
                    for key in blendshapes_dict:
                        if key in prev_blendshapes:
                            prev_blendshapes[key] = SMOOTH_ALPHA * prev_blendshapes[key] + \
                                                   (1.0 - SMOOTH_ALPHA) * blendshapes_dict[key]

            # -----------------------
            # 情绪识别
            # -----------------------
            if emotion_model is not None and scaler is not None:
                try:
                    # 提取特征（优先使用 blendshapes）
                    if prev_blendshapes:
                        features = extract_features_from_blendshapes(prev_blendshapes)
                    else:
                        features = extract_features_from_landmarks(landmarks, w, h)
                    
                    # 标准化
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    
                    # 预测
                    emotion_pred = emotion_model.predict(features_scaled)[0]
                    
                    # 获取置信度（如果模型支持）
                    if hasattr(emotion_model, 'predict_proba'):
                        proba = emotion_model.predict_proba(features_scaled)[0]
                        emotion_confidence = proba.max()
                    else:
                        emotion_confidence = 1.0
                    
                    current_emotion = emotion_pred
                    prev_emotion = current_emotion
                
                except Exception as e:
                    print(f"情绪识别错误: {e}")

            # -----------------------
            # 可视化
            # -----------------------
            # 1. 绘制网格
            overlay = frame.copy()
            max_index = len(landmarks) - 1
            
            for (start, end) in mesh_connections:
                if start <= max_index and end <= max_index:
                    x0 = int(smooth_pts[start][0] * w)
                    y0 = int(smooth_pts[start][1] * h)
                    x1 = int(smooth_pts[end][0] * w)
                    y1 = int(smooth_pts[end][1] * h)
                    cv2.line(overlay, (x0, y0), (x1, y1), (0, 200, 0), 1)
            
            # 轮廓线（根据情绪改变颜色）
            emotion_color = EMOTION_COLORS.get(current_emotion, (255, 255, 255))
            for (start, end) in contour_connections:
                if start <= max_index and end <= max_index:
                    x0 = int(smooth_pts[start][0] * w)
                    y0 = int(smooth_pts[start][1] * h)
                    x1 = int(smooth_pts[end][0] * w)
                    y1 = int(smooth_pts[end][1] * h)
                    cv2.line(overlay, (x0, y0), (x1, y1), emotion_color, 2)
            
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # 2. 显示情绪标签
            label = f"Emotion: {current_emotion.upper()}"
            if emotion_confidence > 0:
                label += f" ({emotion_confidence:.2f})"
            
            # 背景框
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(frame, (10, 50), (20 + text_w, 90 + text_h), emotion_color, -1)
            cv2.putText(frame, label, (15, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

            # 3. 显示关键 BlendShapes
            if prev_blendshapes:
                y_pos = 120
                key_bs = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", 
                         "eyeBlinkLeft", "eyeBlinkRight"]
                
                for bs_name in key_bs:
                    score = prev_blendshapes.get(bs_name, 0.0)
                    text = f"{bs_name}: {score:.2f}"
                    cv2.putText(frame, text, (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_pos += 25

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Real-time Emotion Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

print("程序退出")
