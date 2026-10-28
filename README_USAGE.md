# MediaPipe 面部情绪识别系统使用指南

## 项目概述

本项目基于 MediaPipe Face Landmarker 实现面部关键点提取和情绪识别，包含以下功能：

1. **面部关键点提取**（478个关键点）
2. **BlendShape 表情系数**（52个系数）
3. **网格渲染**（三角网格 + 轮廓线）
4. **情绪识别模型训练**
5. **实时情绪识别**

## 文件说明

- `face_test.py` - 基础测试脚本（静态图片）
- `face_camera.py` - 原始摄像头脚本（基础版）
- `face_camera_enhanced.py` - **增强版**（推荐使用，包含 BlendShapes）
- `train_emotion_model.py` - 情绪识别模型训练脚本
- `face_emotion_realtime.py` - 实时情绪识别（集成版）
- `face_landmarker.task` - MediaPipe 模型文件
- `face_landmarks.csv` - 保存的关键点数据
- `face_landmarks_blendshapes.csv` - 保存的关键点 + BlendShapes 数据

## 使用步骤

### 步骤 1: 测试基础功能

```bash
# 测试静态图片
python face_test.py

# 测试摄像头（基础版）
python face_camera.py
```

### 步骤 2: 使用增强版（接近官方 Demo 效果）

```bash
# 运行增强版（包含 BlendShapes 和完整网格渲染）
python face_camera_enhanced.py
```

**功能：**
- ✅ 478 个面部关键点
- ✅ 完整三角网格渲染
- ✅ 轮廓线高亮
- ✅ 52 个 BlendShape 表情系数实时显示
- ✅ 平滑处理
- ✅ 数据保存到 CSV

**操作：**
- 按 `q` 退出
- 按 `s` 截图

### 步骤 3: 收集训练数据

运行增强版脚本，做出不同的表情：

```bash
python face_camera_enhanced.py
```

**建议收集的表情：**
- neutral（中性）- 10-20秒
- happy（开心）- 10-20秒
- sad（悲伤）- 10-20秒
- angry（生气）- 10-20秒
- surprised（惊讶）- 10-20秒
- disgusted（厌恶）- 10-20秒
- fearful（恐惧）- 10-20秒

数据会自动保存到 `face_landmarks_blendshapes.csv`

### 步骤 4: 标注数据

```bash
# 交互式标注工具
python train_emotion_model.py label
```

程序会显示时间戳，你需要输入对应的情绪标签：
```
时间戳: 1000 ms (0/500)
情绪标签: happy

时间戳: 1030 ms (30/500)
情绪标签: happy

时间戳: 5000 ms (150/500)
情绪标签: sad
...
```

输入 `q` 退出标注。

标注数据会保存到 `emotion_labels.csv`

### 步骤 5: 训练情绪识别模型

```bash
python train_emotion_model.py train
```

程序会：
1. 加载数据和标签
2. 提取特征（优先使用 BlendShapes）
3. 训练多个模型（Random Forest、SVM）
4. 选择最佳模型
5. 保存模型到 `emotion_model.pkl` 和 `emotion_scaler.pkl`

### 步骤 6: 实时情绪识别

```bash
python face_emotion_realtime.py
```

**功能：**
- ✅ 实时面部关键点检测
- ✅ 网格渲染（颜色根据情绪变化）
- ✅ 实时情绪识别和显示
- ✅ 置信度显示
- ✅ 关键 BlendShapes 显示

## 与官方 Demo 的对比

### 官方 Demo 特性
- C++/Python GUI 版本
- OpenGL 渲染三角网格
- GPU 加速
- BlendShape 表情系数
- 平滑处理

### 本项目实现
- ✅ Python 版本
- ✅ OpenCV 渲染（cv2.line 绘制网格）
- ✅ BlendShape 52 个系数
- ✅ EMA 平滑处理
- ✅ 实时 FPS 显示
- ✅ 情绪识别（额外功能）

### 主要区别
1. **渲染方式**：官方用 OpenGL，我们用 OpenCV（性能略低但更简单）
2. **GPU 加速**：官方有，我们暂未实现（可通过 CUDA 优化）
3. **3D 网格**：官方有完整 3D 渲染，我们是 2D 投影

## 性能优化建议

### 1. 提高 FPS
```python
# 降低分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 减少平滑系数（更快响应）
SMOOTH_ALPHA = 0.5
```

### 2. GPU 加速
```python
# 使用 GPU 版本的 MediaPipe（需要 CUDA）
# 或使用 OpenCV 的 CUDA 模块
```

### 3. 多线程处理
```python
# 将渲染和检测分离到不同线程
import threading
```

## 常见问题

### Q1: 摄像头打不开
```bash
# 检查摄像头设备
ls /dev/video*

# 尝试不同的设备 ID
cap = cv2.VideoCapture(1)  # 或 2, 3...
```

### Q2: 模型文件找不到
```bash
# 确保 face_landmarker.task 在当前目录
ls face_landmarker.task

# 或指定完整路径
MODEL_PATH = "/path/to/face_landmarker.task"
```

### Q3: BlendShapes 不显示
确保使用了 `output_face_blendshapes=True`：
```python
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    output_face_blendshapes=True,  # 必须启用
    result_callback=_dummy_callback
)
```

### Q4: 情绪识别不准确
1. 收集更多训练数据（每个情绪至少 100 帧）
2. 标注更准确
3. 尝试不同的模型（SVM、Random Forest、神经网络）
4. 增加特征工程

## 进阶功能

### 1. 使用深度学习模型
```python
# 使用 TensorFlow/PyTorch 训练 CNN
# 输入：BlendShapes (52维) 或 Landmarks (478x3)
# 输出：7 种情绪分类
```

### 2. 添加时序信息
```python
# 使用 LSTM/GRU 处理时间序列
# 输入：连续 N 帧的特征
# 输出：情绪分类
```

### 3. 多人检测
```python
# 修改 NUM_FACES 参数
NUM_FACES = 5

# 遍历所有检测到的人脸
for face_landmarks in result.face_landmarks:
    # 处理每个人脸...
```

## 参考资料

- [MediaPipe 官方文档](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=zh-cn)
- [Face Landmarker API](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [BlendShapes 说明](https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md)

## 联系方式

如有问题，请参考官方文档或提交 Issue。
