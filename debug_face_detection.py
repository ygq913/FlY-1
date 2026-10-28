#!/usr/bin/env python3
"""
debug_face_detection.py
诊断脚本 - 检查 Face Landmarker 是否正常工作
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

MODEL_PATH = "face_landmarker.task"

print("=" * 50)
print("Face Landmarker 诊断工具")
print("=" * 50)

# 1. 检查模型文件
import os
if not os.path.exists(MODEL_PATH):
    print(f"❌ 错误：找不到模型文件 {MODEL_PATH}")
    print("请确保 face_landmarker.task 在当前目录")
    exit(1)
else:
    print(f"✅ 模型文件存在: {MODEL_PATH}")
    print(f"   文件大小: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")

# 2. 测试摄像头
print("\n测试摄像头...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("❌ 无法读取摄像头画面")
    exit(1)

h, w, _ = frame.shape
print(f"✅ 摄像头正常: {w}x{h}")

# 3. 测试 Face Landmarker (IMAGE 模式 - 更简单)
print("\n测试 Face Landmarker (IMAGE 模式)...")
try:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    print("✅ Face Landmarker 初始化成功")
except Exception as e:
    print(f"❌ Face Landmarker 初始化失败: {e}")
    exit(1)

# 4. 实际检测测试
print("\n开始实时检测测试...")
print("请确保你的脸在摄像头前，按 'q' 退出")

detection_count = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 转换为 RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # 检测
    try:
        result = landmarker.detect(mp_image)
        
        if result and result.face_landmarks:
            detection_count += 1
            landmarks = result.face_landmarks[0]
            
            # 画关键点
            for lm in landmarks:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
            
            # 显示信息
            cv2.putText(frame, f"Face Detected! ({len(landmarks)} points)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # BlendShapes
            if result.face_blendshapes:
                blendshapes = result.face_blendshapes[0]
                cv2.putText(frame, f"BlendShapes: {len(blendshapes)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            print(f"✅ 帧 {frame_count}: 检测到人脸 ({len(landmarks)} 关键点)")
        else:
            cv2.putText(frame, "No Face Detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if frame_count % 30 == 0:  # 每30帧打印一次
                print(f"⚠️  帧 {frame_count}: 未检测到人脸")
    
    except Exception as e:
        print(f"❌ 检测错误: {e}")
        break
    
    # 显示统计
    cv2.putText(frame, f"Frames: {frame_count}, Detected: {detection_count}", 
               (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Face Detection Debug", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()

# 5. 总结
print("\n" + "=" * 50)
print("诊断结果:")
print("=" * 50)
print(f"总帧数: {frame_count}")
print(f"检测到人脸的帧数: {detection_count}")
print(f"检测率: {detection_count / frame_count * 100:.1f}%")

if detection_count == 0:
    print("\n❌ 问题：完全没有检测到人脸")
    print("\n可能的原因：")
    print("1. 光线太暗 - 请增加环境光线")
    print("2. 脸部角度不对 - 请正面对着摄像头")
    print("3. 距离太远或太近 - 调整到合适距离")
    print("4. 摄像头分辨率太低 - 尝试更换摄像头")
    print("5. 模型文件损坏 - 重新下载模型")
elif detection_count < frame_count * 0.5:
    print("\n⚠️  警告：检测率较低")
    print("建议：改善光线条件，保持正面对着摄像头")
else:
    print("\n✅ 检测正常！")
