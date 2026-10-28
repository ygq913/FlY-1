#!/usr/bin/env python3
"""
face_landmarker_demo.py

功能：
- Face Landmarker (LIVE_STREAM) 实时检测（478 landmarks）
- 网格渲染（使用 face_mesh connections）
- 简单 EMA 平滑
- CSV 保存 (每帧 flatten 的 x,y,z + timestamp_ms)
- FPS 显示、Avg change 显示
- 简单表情指标（嘴宽、嘴高、眉间高度差）和微表情触发
"""

import time
import csv
import os
import collections
import math

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------
# 配置区（按需调整）
# -----------------------
MODEL_PATH = "face_landmarker.task"
OUTPUT_CSV = "face_landmarks.csv"
NUM_FACES = 1
SMOOTH_ALPHA = 0.6    # EMA 平滑系数 (0..1) 越大越平滑
MICRO_THRESH = 0.012  # 微表情触发阈值（经验值，按需调）
FPS_SMOOTH = 0.9

# 用于嘴、眉索引（参考 FaceMesh 的常用索引）
# 这些索引基于 FaceMesh (468 points) 常用点位，Face Landmarker 的前 468 点通常与 FaceMesh 对齐
# 若你确认 Landmarker 的点位不同，可根据实际映射微调这些索引
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

LEFT_BROW = 55   # 眉眼相对索引（近似）
RIGHT_BROW = 285
LEFT_EYE_TOP = 386
RIGHT_EYE_TOP = 159

# -----------------------
# helpers
# -----------------------
def l2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_distance_pts(p1, p2, w, h):
    return math.hypot((p1[0] - p2[0]) * w, (p1[1] - p2[1]) * h)

# -----------------------
# 初始化 MediaPipe Face Landmarker (LIVE_STREAM)
# -----------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

# 必须给一个回调（即使空实现）以支持 LIVE_STREAM 模式
def _dummy_callback(result, output_image, timestamp_ms):
    # 回调被调用于内部 pipeline 线程；我们这里不在回调里处理
    return

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=NUM_FACES,
    result_callback=_dummy_callback
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# -----------------------
# CSV 文件准备
# -----------------------
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        # 478 points x,y,z
        for i in range(478):
            header += [f"x{i+1}", f"y{i+1}", f"z{i+1}"]
        header += ["timestamp_ms"]
        writer.writerow(header)

# -----------------------
# 打开摄像头
# -----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头，请检查 /dev/video0 或摄像头权限")

# FaceMesh connections（用于画网格）
# 注意：这些连接是为 FaceMesh(468) 设计的，对 478 可能兼容（多数点对齐），若出现 index error 可注释掉
mesh_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION

# 平滑缓存
prev_smoothed = None

# 记录前 N 帧的 avg change 用于画趋势
history_len = 120
avg_history = collections.deque(maxlen=history_len)

# FPS 计算
fps = 0.0
prev_ts = time.time()
start_time = time.time()
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break

        frame_count += 1
        now = time.time()
        timestamp_ms = int((now - start_time) * 1000)

        # FPS smoothing
        dt = now - prev_ts
        prev_ts = now
        if dt > 0:
            inst_fps = 1.0 / dt
            fps = FPS_SMOOTH * fps + (1.0 - FPS_SMOOTH) * inst_fps if fps else inst_fps

        h, w, _ = frame.shape

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # wrap into mediapipe Image (tasks api expects this)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # detect_async 返回 result（在你的版本中这是可用的）
        result = landmarker.detect_async(mp_image, timestamp_ms)

        landmarks = None
        if result and result.face_landmarks:
            landmarks = result.face_landmarks[0]  # 1 face

            # 转 numpy (N,3) 格式 (x,y,z)
            pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

            # 若尚未平滑，先初始化
            if prev_smoothed is None:
                prev_smoothed = pts.copy()
            else:
                # EMA 平滑：新 = alpha * prev + (1-alpha) * current
                prev_smoothed = SMOOTH_ALPHA * prev_smoothed + (1.0 - SMOOTH_ALPHA) * pts

            smooth_pts = prev_smoothed

            # 计算帧间变化量（相对于上一次 smooth）
            # 若 history 中有上一帧，则用差分计算 avg change
            # 这里使用 smooth 前后差异（当前 smooth 与上一 smooth 的 L2）
            # 为了便于显示，计算屏幕像素级距离（乘宽高）
            if 'last_smooth' in locals():
                deltas = np.linalg.norm(smooth_pts - last_smooth, axis=1)
                # convert normalized deltas to pixel deltas (approx)
                pixel_deltas = deltas * np.sqrt(w * h)  # rough scale to pixel; not exact but ok for trend
                avg_delta = float(np.mean(pixel_deltas))
            else:
                avg_delta = 0.0
            last_smooth = smooth_pts.copy()

            avg_history.append(avg_delta)

            # 保存 CSV（保存原始 pts 而非平滑）
            row = pts.flatten().tolist() + [timestamp_ms]
            with open(OUTPUT_CSV, "a", newline="") as f:
                csv.writer(f).writerow(row)

            # 可视化：画网格（连接）
            # 防止 index error：先检查 max index
            max_index = len(landmarks) - 1
            try:
                for (start, end) in mesh_connections:
                    if start <= max_index and end <= max_index:
                        x0, y0 = int(landmarks[start].x * w), int(landmarks[start].y * h)
                        x1, y1 = int(landmarks[end].x * w), int(landmarks[end].y * h)
                        cv2.line(frame, (x0, y0), (x1, y1), (0, 200, 0), 1)
            except Exception:
                # 若 connections 与 478 不兼容可忽略网格
                pass

            # 画点（小圆）
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

            # ------------------------
            # 计算简单表情指标示例
            # ------------------------
            def safe_idx(idx):
                return min(max(idx, 0), len(landmarks)-1)

            try:
                ml = (landmarks[safe_idx(MOUTH_LEFT)].x, landmarks[safe_idx(MOUTH_LEFT)].y)
                mr = (landmarks[safe_idx(MOUTH_RIGHT)].x, landmarks[safe_idx(MOUTH_RIGHT)].y)
                mt = (landmarks[safe_idx(MOUTH_TOP)].x, landmarks[safe_idx(MOUTH_TOP)].y)
                mb = (landmarks[safe_idx(MOUTH_BOTTOM)].x, landmarks[safe_idx(MOUTH_BOTTOM)].y)

                mouth_width_px = compute_distance_pts(ml, mr, w, h)
                mouth_open_px = compute_distance_pts(mt, mb, w, h)

                # 眉高度（相对眼睛）
                lb = (landmarks[safe_idx(LEFT_BROW)].x, landmarks[safe_idx(LEFT_BROW)].y)
                rb = (landmarks[safe_idx(RIGHT_BROW)].x, landmarks[safe_idx(RIGHT_BROW)].y)
                let = (landmarks[safe_idx(LEFT_EYE_TOP)].x, landmarks[safe_idx(LEFT_EYE_TOP)].y)
                ret_ = (landmarks[safe_idx(RIGHT_EYE_TOP)].x, landmarks[safe_idx(RIGHT_EYE_TOP)].y)

                left_brow_eye_dist = compute_distance_pts(lb, let, w, h)
                right_brow_eye_dist = compute_distance_pts(rb, ret_, w, h)

                brow_avg = (left_brow_eye_dist + right_brow_eye_dist) / 2.0

                # 显示数值
                cv2.putText(frame, f"MouthW:{mouth_width_px:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"MouthH:{mouth_open_px:.1f}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Brow:{brow_avg:.1f}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                # 微表情简单触发：若 mouth_open 突然增大
                trigger = False
                if len(avg_history) >= 3:
                    # detect if avg_delta jumps higher than MICRO_THRESH relative to previous
                    if avg_delta > MICRO_THRESH * 100:  # scaled threshold; adjust empirically
                        trigger = True

                if trigger:
                    cv2.putText(frame, "MICRO-EXPR", (w - 160, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            except Exception:
                # 若索引不对，跳过表情指标
                pass

        # ------------------------
        # 边栏：FPS、avg change 历史小图
        # ------------------------
        cv2.putText(frame, f"FPS:{fps:.1f}", (w - 160, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # avg change 显示（最近 history_len 的小条形图）
        hist_h = 60
        hist_w = 200
        hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        if avg_history:
            arr = np.array(avg_history)
            # scale to hist_h
            mx = max(arr.max(), 1e-6)
            norm = (arr / mx) * (hist_h - 4)
            for i, v in enumerate(norm.astype(int)):
                x = int(i * hist_w / history_len)
                cv2.line(hist_img, (x, hist_h), (x, hist_h - v), (0, 120, 255), 1)
        # 将 hist_img 贴到 frame 的右下角
        hh, ww = hist_img.shape[:2]
        frame[-hh-10:-10, -ww-10:-10] = hist_img

        # 显示主窗口
        cv2.imshow("Face Landmarker (LIVE) - Mesh + Metrics", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        landmarker.close()
    except Exception:
        pass

print("退出。")
