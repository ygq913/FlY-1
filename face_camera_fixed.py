#!/usr/bin/env python3
"""
face_camera_fixed.py

修复版 - 使用 IMAGE 模式（更稳定可靠）
功能：
- 478 landmarks 实时检测
- 完整网格渲染
- BlendShape 表情系数显示（含自定义增强特征）
- CSV 保存
"""

import time
import csv
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------
# 配置
# -----------------------
MODEL_PATH = "face_landmarker.task"
OUTPUT_CSV = "face_landmarks_blendshapes.csv"
NUM_FACES = 1
SMOOTH_ALPHA = 0.7

# 原始 BlendShape 名称（52个）
BLENDSHAPE_NAMES = [
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

# 自定义增强特征名称（替代不准确的 BlendShapes）
CUSTOM_FEATURE_NAMES = [
    "custom_mouthCornerLiftLeft",   # 替代 mouthSmileLeft/mouthFrownLeft
    "custom_mouthCornerLiftRight",  # 替代 mouthSmileRight/mouthFrownRight
    "custom_mouthPucker",           # 替代 mouthPucker
    "custom_frownIntensity",        # 替代 browDownLeft/browDownRight
    "custom_browInnerDistance",     # 眉头距离（皱眉检测）
    "custom_noseWrinkle",           # 替代 noseSneerLeft/noseSneerRight
    "custom_mouthOpenRatio",        # 嘴巴张开比例
    "custom_eyeOpenLeft",           # 左眼开合
    "custom_eyeOpenRight",          # 右眼开合
]

# 关键点索引
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
LEFT_BROW_INNER = 107
RIGHT_BROW_INNER = 336
LEFT_BROW_MID = 105
RIGHT_BROW_MID = 334
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
NOSE_BRIDGE_TOP = 6
FACE_LEFT = 234
FACE_RIGHT = 454


def compute_custom_features(pts):
    """从关键点计算自定义特征 - 不做归一化，保留原始比例值"""
    features = {}
    
    # 脸宽（用于归一化）
    face_width = np.sqrt((pts[FACE_LEFT][0] - pts[FACE_RIGHT][0])**2 + 
                         (pts[FACE_LEFT][1] - pts[FACE_RIGHT][1])**2)
    
    # 1. 嘴角上扬/下垂（更准确的微笑/悲伤检测）
    # 正值=上扬(微笑), 负值=下垂(悲伤), 0=平缓
    mouth_center_y = (pts[MOUTH_TOP][1] + pts[MOUTH_BOTTOM][1]) / 2
    left_corner_lift = (mouth_center_y - pts[MOUTH_LEFT][1]) / face_width
    right_corner_lift = (mouth_center_y - pts[MOUTH_RIGHT][1]) / face_width
    
    # 保留原始值（约 -0.05 到 0.05），不做 clip
    features["custom_mouthCornerLiftLeft"] = left_corner_lift
    features["custom_mouthCornerLiftRight"] = right_corner_lift
    
    # 2. 嘟嘴程度（基于嘴宽与脸宽的比例）
    mouth_width = np.sqrt((pts[MOUTH_LEFT][0] - pts[MOUTH_RIGHT][0])**2 + 
                          (pts[MOUTH_LEFT][1] - pts[MOUTH_RIGHT][1])**2)
    # 嘴宽/脸宽 比例，正常约 0.3-0.35，嘟嘴时减小
    features["custom_mouthPucker"] = mouth_width / face_width
    
    # 3. 皱眉强度（眉头距离）
    brow_inner_dist = np.sqrt((pts[LEFT_BROW_INNER][0] - pts[RIGHT_BROW_INNER][0])**2 + 
                              (pts[LEFT_BROW_INNER][1] - pts[RIGHT_BROW_INNER][1])**2)
    # 眉头距离/脸宽 比例，正常约 0.12-0.15，皱眉时减小
    features["custom_frownIntensity"] = brow_inner_dist / face_width
    features["custom_browInnerDistance"] = brow_inner_dist / face_width
    
    # 4. 鼻子皱起（鼻梁到眉心距离）
    brow_center = (pts[LEFT_BROW_INNER] + pts[RIGHT_BROW_INNER]) / 2
    nose_to_brow = np.sqrt((pts[NOSE_BRIDGE_TOP][0] - brow_center[0])**2 + 
                           (pts[NOSE_BRIDGE_TOP][1] - brow_center[1])**2)
    # 鼻梁到眉心距离/脸宽 比例
    features["custom_noseWrinkle"] = nose_to_brow / face_width
    
    # 5. 嘴巴张开比例（嘴高/脸宽）
    mouth_height = np.sqrt((pts[MOUTH_TOP][0] - pts[MOUTH_BOTTOM][0])**2 + 
                           (pts[MOUTH_TOP][1] - pts[MOUTH_BOTTOM][1])**2)
    features["custom_mouthOpenRatio"] = mouth_height / face_width
    
    # 6. 眼睛开合（眼高/脸宽）
    # 正常约 0.03-0.04，睁大时增大，眯眼时减小
    left_eye_height = np.sqrt((pts[LEFT_EYE_TOP][0] - pts[LEFT_EYE_BOTTOM][0])**2 + 
                              (pts[LEFT_EYE_TOP][1] - pts[LEFT_EYE_BOTTOM][1])**2)
    right_eye_height = np.sqrt((pts[RIGHT_EYE_TOP][0] - pts[RIGHT_EYE_BOTTOM][0])**2 + 
                               (pts[RIGHT_EYE_TOP][1] - pts[RIGHT_EYE_BOTTOM][1])**2)
    features["custom_eyeOpenLeft"] = left_eye_height / face_width
    features["custom_eyeOpenRight"] = right_eye_height / face_width
    
    return features

# -----------------------
# 初始化 Face Landmarker (IMAGE 模式 - 更稳定)
# -----------------------
print("初始化 Face Landmarker...")
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,  # 使用 IMAGE 模式
    num_faces=NUM_FACES,
    output_face_blendshapes=True,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.FaceLandmarker.create_from_options(options)
print("✅ 初始化成功")

# -----------------------
# CSV 初始化（包含原始 BlendShapes + 自定义特征）
# -----------------------
csv_exists = os.path.exists(OUTPUT_CSV)
if not csv_exists:
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        # 478 landmarks
        for i in range(478):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        # 52 原始 BlendShapes
        for name in BLENDSHAPE_NAMES:
            header.append(name)
        # 9 自定义增强特征
        for name in CUSTOM_FEATURE_NAMES:
            header.append(name)
        header.append("timestamp_ms")
        writer.writerow(header)
    print(f"✅ 创建 CSV 文件: {OUTPUT_CSV}")
    print(f"   包含: 478 landmarks + 52 BlendShapes + 9 自定义特征")
else:
    print(f"✅ CSV 文件已存在: {OUTPUT_CSV}")

# -----------------------
# 摄像头
# -----------------------
print("打开摄像头...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ 无法打开摄像头")

# 设置分辨率（可选）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ 摄像头已打开")

# 网格连接
mesh_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
contour_connections = mp.solutions.face_mesh.FACEMESH_CONTOURS

# 平滑缓存
prev_smoothed = None
prev_blendshapes = None

# 统计
start_time = time.time()
frame_count = 0
detection_count = 0
fps = 0.0
prev_ts = time.time()

print("\n" + "=" * 50)
print("程序运行中...")
print("=" * 50)
print("按 'q' 退出")
print("按 's' 截图")
print("请确保你的脸在摄像头前，光线充足")
print("=" * 50 + "\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面")
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

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 检测（IMAGE 模式使用 detect 而不是 detect_async）
        try:
            result = landmarker.detect(mp_image)
        except Exception as e:
            print(f"❌ 检测错误: {e}")
            cv2.putText(frame, f"Detection Error: {str(e)[:30]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Face Landmarker - Fixed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        face_detected = False

        if result and result.face_landmarks and len(result.face_landmarks) > 0:
            face_detected = True
            detection_count += 1
            
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
            if result.face_blendshapes and len(result.face_blendshapes) > 0:
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
                        else:
                            prev_blendshapes[key] = blendshapes_dict[key]

            # 计算自定义特征
            custom_features = compute_custom_features(smooth_pts)
            
            # 保存 CSV（landmarks + BlendShapes + 自定义特征）
            row = pts.flatten().tolist()
            # 原始 BlendShapes
            for name in BLENDSHAPE_NAMES:
                row.append(prev_blendshapes.get(name, 0.0) if prev_blendshapes else 0.0)
            # 自定义特征
            for name in CUSTOM_FEATURE_NAMES:
                row.append(custom_features.get(name, 0.0))
            row.append(timestamp_ms)
            
            with open(OUTPUT_CSV, "a", newline="") as f:
                csv.writer(f).writerow(row)

            # -----------------------
            # 可视化：绘制网格
            # -----------------------
            overlay = frame.copy()
            max_index = len(landmarks) - 1
            
            # 1. 绘制三角网格（绿色细线）
            for (start, end) in mesh_connections:
                if start <= max_index and end <= max_index:
                    x0 = int(smooth_pts[start][0] * w)
                    y0 = int(smooth_pts[start][1] * h)
                    x1 = int(smooth_pts[end][0] * w)
                    y1 = int(smooth_pts[end][1] * h)
                    cv2.line(overlay, (x0, y0), (x1, y1), (0, 255, 0), 1)
            
            # 2. 绘制轮廓线（黄色粗线）
            for (start, end) in contour_connections:
                if start <= max_index and end <= max_index:
                    x0 = int(smooth_pts[start][0] * w)
                    y0 = int(smooth_pts[start][1] * h)
                    x1 = int(smooth_pts[end][0] * w)
                    y1 = int(smooth_pts[end][1] * h)
                    cv2.line(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)
            
            # 混合
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # 3. 绘制关键点（小圆点）
            for pt in smooth_pts:
                cx = int(pt[0] * w)
                cy = int(pt[1] * h)
                cv2.circle(frame, (cx, cy), 1, (255, 0, 255), -1)

            # -----------------------
            # 显示 BlendShape 系数 + 自定义特征（右侧面板）
            # -----------------------
            if prev_blendshapes:
                panel_w = 320
                panel_h = h
                panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                
                y_offset = 20
                
                # 标题：自定义增强特征（更准确）
                cv2.putText(panel, "Custom Features (Enhanced)", (5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                y_offset += 20
                
                # 显示自定义特征
                custom_display = [
                    ("MouthCornerL", "custom_mouthCornerLiftLeft"),
                    ("MouthCornerR", "custom_mouthCornerLiftRight"),
                    ("MouthPucker*", "custom_mouthPucker"),
                    ("FrownIntensity*", "custom_frownIntensity"),
                    ("BrowInnerDist", "custom_browInnerDistance"),
                    ("NoseWrinkle*", "custom_noseWrinkle"),
                    ("MouthOpenRatio", "custom_mouthOpenRatio"),
                    ("EyeOpenL", "custom_eyeOpenLeft"),
                    ("EyeOpenR", "custom_eyeOpenRight"),
                ]
                
                for label, key in custom_display:
                    score = custom_features.get(key, 0.0)
                    # 根据不同特征调整显示比例
                    if "mouthCornerLift" in key:
                        # 嘴角：-0.05 到 0.05，中心点在 0
                        bar_len = int((score + 0.05) / 0.1 * 180)
                    elif "mouthPucker" in key:
                        # 嘴宽比例：0.2-0.4，正常约0.3
                        bar_len = int(score / 0.5 * 180)
                    elif "frown" in key or "browInner" in key:
                        # 眉头距离：0.08-0.18
                        bar_len = int(score / 0.2 * 180)
                    elif "eyeOpen" in key:
                        # 眼睛开合：0.02-0.06
                        bar_len = int(score / 0.08 * 180)
                    else:
                        bar_len = int(score / 0.2 * 180)
                    
                    bar_len = max(0, min(180, bar_len))
                    # 青色表示自定义特征
                    cv2.rectangle(panel, (5, y_offset), (5 + bar_len, y_offset + 12), (255, 255, 0), -1)
                    text = f"{label}: {score:.4f}"
                    cv2.putText(panel, text, (5, y_offset + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    y_offset += 18
                
                y_offset += 10
                
                # 标题：原始 BlendShapes
                cv2.putText(panel, "Original BlendShapes", (5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                y_offset += 20
                
                # 显示原始 BlendShapes
                important_bs = [
                    "jawOpen", "mouthSmileLeft", "mouthSmileRight", 
                    "mouthFrownLeft", "mouthFrownRight",
                    "eyeBlinkLeft", "eyeBlinkRight",
                    "browInnerUp", "browDownLeft", "browDownRight",
                    "mouthPucker", "eyeWideLeft", "eyeWideRight",
                    "noseSneerLeft", "noseSneerRight",
                ]
                
                for bs_name in important_bs:
                    if y_offset > h - 30:
                        break
                    score = prev_blendshapes.get(bs_name, 0.0)
                    bar_len = int(score * 180)
                    color = (0, 255, 0) if score < 0.5 else (0, 255, 255)
                    cv2.rectangle(panel, (5, y_offset), (5 + bar_len, y_offset + 12), color, -1)
                    text = f"{bs_name[:14]}: {score:.2f}"
                    cv2.putText(panel, text, (5, y_offset + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    y_offset += 18
                
                # 合并
                frame = np.hstack([frame, panel])
            
            # 状态显示
            cv2.putText(frame, "Face Detected!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"{len(landmarks)} landmarks", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        else:
            # 未检测到人脸
            cv2.putText(frame, "No Face Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Please face the camera", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # FPS 和统计
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Detected: {detection_count} ({detection_count/frame_count*100:.1f}%)", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Face Landmarker - Fixed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n退出程序...")
            break
        elif key == ord("s"):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"📸 截图保存: {filename}")

except KeyboardInterrupt:
    print("\n程序被中断")
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

# 最终统计
print("\n" + "=" * 50)
print("运行统计:")
print("=" * 50)
print(f"总帧数: {frame_count}")
print(f"检测到人脸的帧数: {detection_count}")
print(f"检测率: {detection_count / frame_count * 100:.1f}%" if frame_count > 0 else "N/A")
print(f"CSV 文件: {OUTPUT_CSV}")
print("=" * 50)
