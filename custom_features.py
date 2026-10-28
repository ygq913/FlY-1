#!/usr/bin/env python3
"""
custom_features.py

自定义面部特征提取 - 基于 478 个关键点的几何计算
解决 BlendShapes 不准确的问题
"""

import numpy as np

# =====================================================
# MediaPipe Face Landmarker 关键点索引
# =====================================================

# 嘴部关键点
MOUTH_LEFT = 61          # 左嘴角
MOUTH_RIGHT = 291        # 右嘴角
MOUTH_TOP = 13           # 上唇中心
MOUTH_BOTTOM = 14        # 下唇中心
UPPER_LIP_TOP = 0        # 上唇上边缘
LOWER_LIP_BOTTOM = 17    # 下唇下边缘

# 眉毛关键点
LEFT_BROW_INNER = 107    # 左眉内侧（眉头）
LEFT_BROW_OUTER = 70     # 左眉外侧
LEFT_BROW_MID = 105      # 左眉中间
RIGHT_BROW_INNER = 336   # 右眉内侧（眉头）
RIGHT_BROW_OUTER = 300   # 右眉外侧
RIGHT_BROW_MID = 334     # 右眉中间

# 眼睛关键点
LEFT_EYE_TOP = 159       # 左眼上眼睑
LEFT_EYE_BOTTOM = 145    # 左眼下眼睑
LEFT_EYE_INNER = 133     # 左眼内角
LEFT_EYE_OUTER = 33      # 左眼外角
RIGHT_EYE_TOP = 386      # 右眼上眼睑
RIGHT_EYE_BOTTOM = 374   # 右眼下眼睑
RIGHT_EYE_INNER = 362    # 右眼内角
RIGHT_EYE_OUTER = 263    # 右眼外角

# 鼻子关键点
NOSE_TIP = 1             # 鼻尖
NOSE_BRIDGE_TOP = 6      # 鼻梁顶部
NOSE_BRIDGE_MID = 197    # 鼻梁中部
NOSE_LEFT = 129          # 鼻子左侧
NOSE_RIGHT = 358         # 鼻子右侧

# 脸部轮廓
FACE_LEFT = 234          # 左脸颊
FACE_RIGHT = 454         # 右脸颊
CHIN = 152               # 下巴
FOREHEAD_MID = 10        # 额头中心

# =====================================================
# 辅助函数
# =====================================================

def distance_3d(p1, p2):
    """计算两点之间的3D距离"""
    return np.sqrt(np.sum((p1 - p2) ** 2))

def distance_2d(p1, p2):
    """计算两点之间的2D距离（忽略z）"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def angle_between(v1, v2):
    """计算两个向量之间的角度（弧度）"""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos_angle, -1, 1))

def normalize_by_face_width(value, face_width):
    """用脸宽归一化数值"""
    return value / (face_width + 1e-8)

# =====================================================
# 自定义特征提取
# =====================================================

def extract_custom_features(landmarks):
    """
    从 478 个关键点提取自定义特征
    
    参数:
        landmarks: numpy array, shape (478, 3) 或 list of landmarks
    
    返回:
        dict: 特征名 -> 特征值
    """
    
    # 转换为 numpy array
    if not isinstance(landmarks, np.ndarray):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    else:
        pts = landmarks
    
    features = {}
    
    # -----------------------
    # 基准：脸部宽度（用于归一化）
    # -----------------------
    face_width = distance_2d(pts[FACE_LEFT], pts[FACE_RIGHT])
    face_height = distance_2d(pts[FOREHEAD_MID], pts[CHIN])
    
    features['face_width'] = face_width
    features['face_height'] = face_height
    
    # =====================================================
    # 1. 嘴部特征（改进版）
    # =====================================================
    
    # 嘴宽
    mouth_width = distance_2d(pts[MOUTH_LEFT], pts[MOUTH_RIGHT])
    features['mouth_width'] = normalize_by_face_width(mouth_width, face_width)
    
    # 嘴高（张嘴程度）
    mouth_height = distance_2d(pts[MOUTH_TOP], pts[MOUTH_BOTTOM])
    features['mouth_height'] = normalize_by_face_width(mouth_height, face_width)
    
    # 嘴部纵横比
    features['mouth_aspect_ratio'] = mouth_height / (mouth_width + 1e-8)
    
    # 嘴角高度差（微笑/悲伤的关键指标）
    # 正值 = 嘴角上扬（微笑），负值 = 嘴角下垂（悲伤）
    mouth_center_y = (pts[MOUTH_TOP][1] + pts[MOUTH_BOTTOM][1]) / 2
    left_corner_offset = mouth_center_y - pts[MOUTH_LEFT][1]
    right_corner_offset = mouth_center_y - pts[MOUTH_RIGHT][1]
    
    features['mouth_corner_lift_left'] = normalize_by_face_width(left_corner_offset, face_width)
    features['mouth_corner_lift_right'] = normalize_by_face_width(right_corner_offset, face_width)
    features['mouth_corner_lift_avg'] = (features['mouth_corner_lift_left'] + features['mouth_corner_lift_right']) / 2
    
    # 嘴唇厚度
    lip_thickness = distance_2d(pts[UPPER_LIP_TOP], pts[LOWER_LIP_BOTTOM])
    features['lip_thickness'] = normalize_by_face_width(lip_thickness, face_width)
    
    # 嘴部收紧程度（嘟嘴）- 用嘴宽和嘴高的比例
    features['mouth_pucker'] = 1.0 - (mouth_width / (face_width * 0.4 + 1e-8))
    
    # =====================================================
    # 2. 眉毛特征（改进版 - 解决皱眉识别问题）
    # =====================================================
    
    # 眉毛高度（相对于眼睛）
    left_brow_height = pts[LEFT_EYE_TOP][1] - pts[LEFT_BROW_MID][1]
    right_brow_height = pts[RIGHT_EYE_TOP][1] - pts[RIGHT_BROW_MID][1]
    
    features['left_brow_height'] = normalize_by_face_width(left_brow_height, face_width)
    features['right_brow_height'] = normalize_by_face_width(right_brow_height, face_width)
    features['brow_height_avg'] = (features['left_brow_height'] + features['right_brow_height']) / 2
    
    # 眉头距离（皱眉的关键指标！）
    # 皱眉时眉头会靠近
    brow_inner_distance = distance_2d(pts[LEFT_BROW_INNER], pts[RIGHT_BROW_INNER])
    features['brow_inner_distance'] = normalize_by_face_width(brow_inner_distance, face_width)
    
    # 眉头相对于鼻梁的位置（更准确的皱眉检测）
    nose_bridge_x = pts[NOSE_BRIDGE_TOP][0]
    left_brow_inner_offset = pts[LEFT_BROW_INNER][0] - nose_bridge_x
    right_brow_inner_offset = nose_bridge_x - pts[RIGHT_BROW_INNER][0]
    
    features['left_brow_inner_offset'] = normalize_by_face_width(left_brow_inner_offset, face_width)
    features['right_brow_inner_offset'] = normalize_by_face_width(right_brow_inner_offset, face_width)
    
    # 皱眉程度（眉头越近，值越大）
    # 基准值约为 0.15，皱眉时会减小
    baseline_brow_distance = face_width * 0.15
    features['frown_intensity'] = max(0, (baseline_brow_distance - brow_inner_distance) / baseline_brow_distance)
    
    # 眉毛倾斜角度（内侧到外侧）
    left_brow_vector = pts[LEFT_BROW_OUTER] - pts[LEFT_BROW_INNER]
    right_brow_vector = pts[RIGHT_BROW_INNER] - pts[RIGHT_BROW_OUTER]
    
    # 眉毛角度（正值=眉毛上扬，负值=眉毛下垂）
    features['left_brow_angle'] = np.arctan2(left_brow_vector[1], left_brow_vector[0])
    features['right_brow_angle'] = np.arctan2(right_brow_vector[1], right_brow_vector[0])
    
    # 眉头高度差（不对称皱眉）
    features['brow_inner_height_diff'] = pts[LEFT_BROW_INNER][1] - pts[RIGHT_BROW_INNER][1]
    
    # =====================================================
    # 3. 眼睛特征
    # =====================================================
    
    # 眼睛开合度
    left_eye_height = distance_2d(pts[LEFT_EYE_TOP], pts[LEFT_EYE_BOTTOM])
    right_eye_height = distance_2d(pts[RIGHT_EYE_TOP], pts[RIGHT_EYE_BOTTOM])
    left_eye_width = distance_2d(pts[LEFT_EYE_INNER], pts[LEFT_EYE_OUTER])
    right_eye_width = distance_2d(pts[RIGHT_EYE_INNER], pts[RIGHT_EYE_OUTER])
    
    features['left_eye_openness'] = left_eye_height / (left_eye_width + 1e-8)
    features['right_eye_openness'] = right_eye_height / (right_eye_width + 1e-8)
    features['eye_openness_avg'] = (features['left_eye_openness'] + features['right_eye_openness']) / 2
    
    # 眼睛睁大程度（惊讶）
    features['left_eye_wide'] = normalize_by_face_width(left_eye_height, face_width)
    features['right_eye_wide'] = normalize_by_face_width(right_eye_height, face_width)
    
    # =====================================================
    # 4. 鼻子特征（改进版 - 用鼻梁位置代替鼻翼）
    # =====================================================
    
    # 鼻梁水平位置（相对于脸部中心）
    face_center_x = (pts[FACE_LEFT][0] + pts[FACE_RIGHT][0]) / 2
    nose_bridge_offset_x = pts[NOSE_BRIDGE_MID][0] - face_center_x
    features['nose_bridge_offset_x'] = normalize_by_face_width(nose_bridge_offset_x, face_width)
    
    # 鼻尖位置变化
    nose_tip_offset_x = pts[NOSE_TIP][0] - face_center_x
    features['nose_tip_offset_x'] = normalize_by_face_width(nose_tip_offset_x, face_width)
    
    # 鼻子皱起程度（通过鼻梁和眉心的距离变化）
    nose_to_brow_distance = distance_2d(pts[NOSE_BRIDGE_TOP], 
                                        (pts[LEFT_BROW_INNER] + pts[RIGHT_BROW_INNER]) / 2)
    features['nose_scrunch'] = normalize_by_face_width(nose_to_brow_distance, face_width)
    
    # =====================================================
    # 5. 综合表情特征
    # =====================================================
    
    # 微笑指数（嘴角上扬 + 眼睛微眯）
    smile_mouth = features['mouth_corner_lift_avg']
    smile_eyes = 1.0 - features['eye_openness_avg']  # 眼睛微眯
    features['smile_index'] = (smile_mouth * 0.7 + smile_eyes * 0.3)
    
    # 悲伤指数（嘴角下垂 + 眉毛内侧上扬）
    sad_mouth = -features['mouth_corner_lift_avg']
    sad_brow = features['brow_height_avg']
    features['sad_index'] = max(0, sad_mouth * 0.6 + sad_brow * 0.4)
    
    # 生气指数（皱眉 + 嘴唇紧闭）
    angry_frown = features['frown_intensity']
    angry_mouth = 1.0 - features['mouth_aspect_ratio']  # 嘴唇紧闭
    features['angry_index'] = (angry_frown * 0.7 + angry_mouth * 0.3)
    
    # 惊讶指数（眼睛睁大 + 嘴巴张开 + 眉毛上扬）
    surprise_eyes = features['eye_openness_avg']
    surprise_mouth = features['mouth_aspect_ratio']
    surprise_brow = features['brow_height_avg']
    features['surprise_index'] = (surprise_eyes * 0.4 + surprise_mouth * 0.3 + surprise_brow * 0.3)
    
    # 厌恶指数（鼻子皱起 + 上唇上扬）
    features['disgust_index'] = features['nose_scrunch'] * 0.5 + features['mouth_corner_lift_avg'] * 0.5
    
    return features


def get_feature_names():
    """返回所有特征名列表"""
    return [
        # 基准
        'face_width', 'face_height',
        # 嘴部
        'mouth_width', 'mouth_height', 'mouth_aspect_ratio',
        'mouth_corner_lift_left', 'mouth_corner_lift_right', 'mouth_corner_lift_avg',
        'lip_thickness', 'mouth_pucker',
        # 眉毛
        'left_brow_height', 'right_brow_height', 'brow_height_avg',
        'brow_inner_distance', 'left_brow_inner_offset', 'right_brow_inner_offset',
        'frown_intensity', 'left_brow_angle', 'right_brow_angle', 'brow_inner_height_diff',
        # 眼睛
        'left_eye_openness', 'right_eye_openness', 'eye_openness_avg',
        'left_eye_wide', 'right_eye_wide',
        # 鼻子
        'nose_bridge_offset_x', 'nose_tip_offset_x', 'nose_scrunch',
        # 综合
        'smile_index', 'sad_index', 'angry_index', 'surprise_index', 'disgust_index'
    ]


# =====================================================
# 测试
# =====================================================
if __name__ == "__main__":
    print("自定义特征提取模块")
    print("=" * 50)
    print(f"共 {len(get_feature_names())} 个特征:")
    for i, name in enumerate(get_feature_names(), 1):
        print(f"  {i:2d}. {name}")
