#!/usr/bin/env python3
"""
label_emotions.py

简化版情绪标注工具
支持两种标注方式：
1. 按时间段批量标注
2. 逐帧标注
"""

import pandas as pd
import os

CSV_FILE = "face_landmarks_blendshapes.csv"
LABELS_FILE = "emotion_labels.csv"

EMOTIONS = {
    '1': 'neutral',
    '2': 'happy', 
    '3': 'sad',
    '4': 'angry',
    '5': 'surprised',
    '6': 'disgusted',
    '7': 'fearful'
}

def show_menu():
    print("\n" + "=" * 50)
    print("情绪标注工具")
    print("=" * 50)
    print("\n可用情绪:")
    for key, emotion in EMOTIONS.items():
        emoji = {'neutral': '😐', 'happy': '😊', 'sad': '😢', 
                 'angry': '😠', 'surprised': '😮', 'disgusted': '🤢', 
                 'fearful': '😨'}
        print(f"  {key}. {emotion} {emoji.get(emotion, '')}")
    print("\n命令:")
    print("  r - 按时间范围标注（推荐）")
    print("  s - 查看当前标注统计")
    print("  c - 清除所有标注")
    print("  q - 退出")
    print("=" * 50)

def load_data():
    if not os.path.exists(CSV_FILE):
        print(f"❌ 错误：找不到数据文件 {CSV_FILE}")
        print("请先运行 face_camera_fixed.py 采集数据")
        return None
    
    df = pd.read_csv(CSV_FILE)
    print(f"✅ 加载了 {len(df)} 条数据")
    
    if 'timestamp_ms' in df.columns:
        total_time = df['timestamp_ms'].max() / 1000
        print(f"   总时长: {total_time:.1f} 秒")
    
    return df

def load_labels():
    if os.path.exists(LABELS_FILE):
        return pd.read_csv(LABELS_FILE)
    else:
        return pd.DataFrame(columns=['timestamp_ms', 'emotion'])

def save_labels(labels_df):
    labels_df.to_csv(LABELS_FILE, index=False)
    print(f"✅ 标注已保存到 {LABELS_FILE}")

def show_stats(df, labels_df):
    print("\n📊 标注统计:")
    print("-" * 30)
    
    if len(labels_df) == 0:
        print("暂无标注数据")
        return
    
    print(f"总标注数: {len(labels_df)}")
    print(f"数据覆盖率: {len(labels_df) / len(df) * 100:.1f}%")
    print("\n各情绪分布:")
    counts = labels_df['emotion'].value_counts()
    for emotion, count in counts.items():
        print(f"  {emotion}: {count} ({count/len(labels_df)*100:.1f}%)")

def label_by_range(df, labels_df):
    """按时间范围批量标注"""
    print("\n📝 按时间范围标注")
    print("-" * 30)
    
    total_time = df['timestamp_ms'].max() / 1000
    print(f"数据总时长: {total_time:.1f} 秒")
    print("\n请输入时间范围和情绪标签")
    print("格式: 开始秒数 结束秒数 情绪编号")
    print("例如: 0 15 1  (表示 0-15秒 标注为 neutral)")
    print("输入 'done' 完成标注\n")
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() == 'done':
                break
            
            parts = user_input.split()
            if len(parts) != 3:
                print("❌ 格式错误，请输入: 开始秒数 结束秒数 情绪编号")
                continue
            
            start_sec = float(parts[0])
            end_sec = float(parts[1])
            emotion_key = parts[2]
            
            if emotion_key not in EMOTIONS:
                print(f"❌ 无效的情绪编号，请使用 1-7")
                continue
            
            emotion = EMOTIONS[emotion_key]
            
            # 转换为毫秒
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            
            # 筛选时间范围内的数据
            mask = (df['timestamp_ms'] >= start_ms) & (df['timestamp_ms'] <= end_ms)
            timestamps = df.loc[mask, 'timestamp_ms'].values
            
            if len(timestamps) == 0:
                print(f"❌ 该时间范围内没有数据")
                continue
            
            # 添加标注
            new_labels = pd.DataFrame({
                'timestamp_ms': timestamps,
                'emotion': [emotion] * len(timestamps)
            })
            
            # 移除已有的重复标注
            labels_df = labels_df[~labels_df['timestamp_ms'].isin(timestamps)]
            labels_df = pd.concat([labels_df, new_labels], ignore_index=True)
            
            print(f"✅ 已标注 {len(timestamps)} 帧为 {emotion} ({start_sec}s - {end_sec}s)")
            
        except ValueError as e:
            print(f"❌ 输入错误: {e}")
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    return labels_df

def main():
    df = load_data()
    if df is None:
        return
    
    labels_df = load_labels()
    print(f"已有 {len(labels_df)} 条标注")
    
    while True:
        show_menu()
        choice = input("\n请选择操作: ").strip().lower()
        
        if choice == 'r':
            labels_df = label_by_range(df, labels_df)
            save_labels(labels_df)
        
        elif choice == 's':
            show_stats(df, labels_df)
        
        elif choice == 'c':
            confirm = input("确定要清除所有标注吗？(y/n): ").strip().lower()
            if confirm == 'y':
                labels_df = pd.DataFrame(columns=['timestamp_ms', 'emotion'])
                save_labels(labels_df)
                print("✅ 已清除所有标注")
        
        elif choice == 'q':
            print("再见！")
            break
        
        else:
            print("❌ 无效选择")

if __name__ == "__main__":
    main()
