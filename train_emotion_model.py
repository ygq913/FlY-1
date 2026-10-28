#!/usr/bin/env python3
"""
train_emotion_model.py

基于保存的 CSV 数据训练情绪识别模型
支持的情绪：neutral, happy, sad, angry, surprised, disgusted, fearful
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# -----------------------
# 配置
# -----------------------
CSV_FILE = "face_landmarks_blendshapes.csv"  # 或 "face_landmarks.csv"
MODEL_OUTPUT = "emotion_model.pkl"
SCALER_OUTPUT = "emotion_scaler.pkl"

# 情绪标签（需要手动标注数据）
EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "disgusted", "fearful"]

def extract_features_from_landmarks(df):
    """
    从 landmarks 提取特征
    特征工程：计算关键点之间的距离、角度等
    """
    features = []
    
    # 假设 CSV 格式：x0, y0, z0, x1, y1, z1, ..., x477, y477, z477
    num_landmarks = 478
    
    for idx, row in df.iterrows():
        # 提取所有 landmarks
        landmarks = []
        for i in range(num_landmarks):
            x = row.get(f'x{i}', 0)
            y = row.get(f'y{i}', 0)
            z = row.get(f'z{i}', 0)
            landmarks.append([x, y, z])
        
        landmarks = np.array(landmarks)
        
        # 特征1: 关键点间距离（选择重要的点）
        # 嘴部
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
        
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
        mouth_aspect_ratio = mouth_height / (mouth_width + 1e-6)
        
        # 眼睛
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        eye_avg_height = (left_eye_height + right_eye_height) / 2
        
        # 眉毛
        left_brow = landmarks[55]
        right_brow = landmarks[285]
        brow_distance = np.linalg.norm(left_brow - right_brow)
        
        # 鼻子
        nose_tip = landmarks[1]
        nose_bridge = landmarks[6]
        nose_length = np.linalg.norm(nose_tip - nose_bridge)
        
        # 脸部宽度
        left_face = landmarks[234]
        right_face = landmarks[454]
        face_width = np.linalg.norm(left_face - right_face)
        
        # 组合特征
        feature_vector = [
            mouth_width, mouth_height, mouth_aspect_ratio,
            left_eye_height, right_eye_height, eye_avg_height,
            brow_distance, nose_length, face_width,
            # 可以添加更多特征...
        ]
        
        features.append(feature_vector)
    
    return np.array(features)

def extract_features_from_blendshapes(df):
    """
    从 BlendShapes 提取特征（如果 CSV 包含 blendshapes）
    """
    blendshape_cols = [col for col in df.columns if col not in ['timestamp_ms'] 
                       and not col.startswith('x') and not col.startswith('y') 
                       and not col.startswith('z')]
    
    if blendshape_cols:
        return df[blendshape_cols].values
    else:
        return None

def create_sample_labeled_data():
    """
    创建示例标注数据（实际使用时需要手动标注）
    """
    print("=" * 50)
    print("注意：需要手动标注数据！")
    print("=" * 50)
    print("\n请按以下步骤准备训练数据：")
    print("1. 运行 face_camera_enhanced.py 收集不同表情的数据")
    print("2. 在收集时，对每个表情录制 10-20 秒")
    print("3. 创建标注文件 'emotion_labels.csv'，格式：")
    print("   timestamp_ms,emotion")
    print("   1000,happy")
    print("   2000,happy")
    print("   3000,sad")
    print("   ...")
    print("\n或者使用下面的交互式标注工具")
    print("=" * 50)

def interactive_labeling():
    """
    交互式标注工具
    """
    if not os.path.exists(CSV_FILE):
        print(f"错误：找不到 {CSV_FILE}")
        return
    
    df = pd.read_csv(CSV_FILE)
    print(f"加载了 {len(df)} 条数据")
    
    labels_file = "emotion_labels.csv"
    
    if os.path.exists(labels_file):
        labels_df = pd.read_csv(labels_file)
        print(f"已有 {len(labels_df)} 条标注")
    else:
        labels_df = pd.DataFrame(columns=['timestamp_ms', 'emotion'])
    
    print("\n开始标注（输入情绪标签，或 'q' 退出）")
    print(f"可用情绪: {', '.join(EMOTIONS)}")
    
    # 每隔 N 帧标注一次
    step = 30  # 每30帧标注一次
    
    for i in range(0, len(df), step):
        timestamp = df.iloc[i]['timestamp_ms']
        
        # 检查是否已标注
        if timestamp in labels_df['timestamp_ms'].values:
            continue
        
        print(f"\n时间戳: {timestamp} ms ({i}/{len(df)})")
        emotion = input("情绪标签: ").strip().lower()
        
        if emotion == 'q':
            break
        
        if emotion in EMOTIONS:
            new_row = pd.DataFrame({'timestamp_ms': [timestamp], 'emotion': [emotion]})
            labels_df = pd.concat([labels_df, new_row], ignore_index=True)
            labels_df.to_csv(labels_file, index=False)
            print(f"已保存标注: {emotion}")
        else:
            print(f"无效情绪，请使用: {', '.join(EMOTIONS)}")
    
    print(f"\n标注完成！共 {len(labels_df)} 条")

def train_model():
    """
    训练情绪识别模型
    """
    if not os.path.exists(CSV_FILE):
        print(f"错误：找不到 {CSV_FILE}")
        return
    
    labels_file = "emotion_labels.csv"
    if not os.path.exists(labels_file):
        print(f"错误：找不到标注文件 {labels_file}")
        print("请先运行标注工具")
        return
    
    # 加载数据
    df = pd.read_csv(CSV_FILE)
    labels_df = pd.read_csv(labels_file)
    
    print(f"数据: {len(df)} 条")
    print(f"标注: {len(labels_df)} 条")
    
    # 合并数据和标签
    merged = pd.merge(df, labels_df, on='timestamp_ms', how='inner')
    print(f"合并后: {len(merged)} 条")
    
    if len(merged) < 10:
        print("错误：标注数据太少，至少需要 10 条")
        return
    
    # 提取特征
    print("\n提取特征...")
    
    # 尝试使用 blendshapes
    X_blend = extract_features_from_blendshapes(merged)
    
    if X_blend is not None and X_blend.shape[1] > 0:
        print(f"使用 BlendShapes 特征: {X_blend.shape[1]} 维")
        X = X_blend
    else:
        print("使用 Landmarks 几何特征")
        X = extract_features_from_landmarks(merged)
    
    y = merged['emotion'].values
    
    print(f"特征维度: {X.shape}")
    print(f"标签分布:\n{pd.Series(y).value_counts()}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型（尝试多个模型）
    print("\n训练模型...")
    
    # Random Forest
    print("1. Random Forest")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_score = rf_model.score(X_test_scaled, y_test)
    print(f"   准确率: {rf_score:.3f}")
    
    # SVM
    print("2. SVM")
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_score = svm_model.score(X_test_scaled, y_test)
    print(f"   准确率: {svm_score:.3f}")
    
    # 选择最佳模型
    if rf_score >= svm_score:
        best_model = rf_model
        best_score = rf_score
        model_name = "Random Forest"
    else:
        best_model = svm_model
        best_score = svm_score
        model_name = "SVM"
    
    print(f"\n最佳模型: {model_name} (准确率: {best_score:.3f})")
    
    # 详细评估
    y_pred = best_model.predict(X_test_scaled)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    # 保存模型
    joblib.dump(best_model, MODEL_OUTPUT)
    joblib.dump(scaler, SCALER_OUTPUT)
    print(f"\n模型已保存: {MODEL_OUTPUT}")
    print(f"标准化器已保存: {SCALER_OUTPUT}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python train_emotion_model.py label   # 交互式标注")
        print("  python train_emotion_model.py train   # 训练模型")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "label":
        interactive_labeling()
    elif command == "train":
        train_model()
    else:
        print(f"未知命令: {command}")
