#!/usr/bin/env python3
"""
train_and_test.py

训练情绪识别模型 - 简化版
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 文件路径
CSV_FILE = "face_landmarks_blendshapes.csv"
LABELS_FILE = "emotion_labels.csv"
MODEL_FILE = "emotion_model.pkl"
SCALER_FILE = "emotion_scaler.pkl"

# BlendShape 特征名（用于训练）
BLENDSHAPE_FEATURES = [
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

def train_model():
    print("=" * 50)
    print("情绪识别模型训练")
    print("=" * 50)
    
    # 1. 检查文件
    if not os.path.exists(CSV_FILE):
        print(f"❌ 找不到数据文件: {CSV_FILE}")
        return False
    
    if not os.path.exists(LABELS_FILE):
        print(f"❌ 找不到标注文件: {LABELS_FILE}")
        print("请先运行: python label_emotions.py")
        return False
    
    # 2. 加载数据
    print("\n📂 加载数据...")
    df = pd.read_csv(CSV_FILE)
    labels_df = pd.read_csv(LABELS_FILE)
    
    print(f"   数据总量: {len(df)} 帧")
    print(f"   标注总量: {len(labels_df)} 帧")
    
    # 3. 合并数据和标签
    merged = pd.merge(df, labels_df, on='timestamp_ms', how='inner')
    print(f"   匹配数据: {len(merged)} 帧")
    
    if len(merged) < 50:
        print(f"❌ 标注数据太少（{len(merged)}），至少需要 50 条")
        return False
    
    # 4. 检查标签分布
    print("\n📊 标签分布:")
    label_counts = merged['emotion'].value_counts()
    for emotion, count in label_counts.items():
        print(f"   {emotion}: {count}")
    
    if len(label_counts) < 2:
        print("❌ 至少需要 2 种不同的情绪标签")
        return False
    
    # 5. 提取特征
    print("\n🔧 提取特征...")
    
    # 检查哪些 BlendShape 列存在
    available_features = [f for f in BLENDSHAPE_FEATURES if f in merged.columns]
    
    if len(available_features) == 0:
        print("❌ 找不到 BlendShape 特征列")
        print("请确保 CSV 文件包含 BlendShape 数据")
        return False
    
    print(f"   使用 {len(available_features)} 个 BlendShape 特征")
    
    X = merged[available_features].values
    y = merged['emotion'].values
    
    # 6. 划分训练集和测试集
    print("\n📊 划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   训练集: {len(X_train)} 样本")
    print(f"   测试集: {len(X_test)} 样本")
    
    # 7. 标准化
    print("\n⚙️  标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 8. 训练模型
    print("\n🚀 训练模型...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # 9. 评估
    print("\n📈 模型评估:")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n   准确率: {accuracy * 100:.1f}%")
    
    print("\n   详细报告:")
    print(classification_report(y_test, y_pred))
    
    # 10. 保存模型
    print("\n💾 保存模型...")
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    # 保存特征名
    with open("feature_names.txt", "w") as f:
        f.write("\n".join(available_features))
    
    print(f"   模型: {MODEL_FILE}")
    print(f"   标准化器: {SCALER_FILE}")
    print(f"   特征名: feature_names.txt")
    
    # 11. 特征重要性
    print("\n🔍 最重要的特征:")
    importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "=" * 50)
    print("✅ 训练完成！")
    print("=" * 50)
    print("\n下一步：运行 python face_emotion_realtime_v2.py 进行实时测试")
    
    return True

if __name__ == "__main__":
    train_model()
