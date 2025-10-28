# 中文对话情绪识别训练指南

## 📋 概述

这个项目基于DialogueGCN模型，专门针对中文对话进行情绪识别训练，适配datapre文件夹的数据格式。

## 🎯 情绪标签映射

严格按照用户指定的情绪标签映射：

| 原始标签                                                  | 映射到       | 数值 | 描述 |
| --------------------------------------------------------- | ------------ | ---- | ---- |
| `relaxed`, `happy`, `grateful`, `positive-other`          | **happy**    | 1    | 开心 |
| `neutral`                                                 | **neutral**  | 0    | 中性 |
| `astonished`                                              | **surprise** | 5    | 惊讶 |
| `depress`, `fear`, `negative-other`, `sadness`, `worried` | **sad**      | 2    | 悲伤 |
| `anger`, `disgust`                                        | **angry**    | 3    | 愤怒 |

**注意**: 只支持这5种情绪类别，没有其他类别。

## 🚀 快速开始

### 1. 准备数据

确保您已经有了datapre文件夹处理后的pkl数据文件。

### 2. 开始训练

```bash
# 使用GPU训练（推荐）
python run_chinese_training.py --data_path ../datapre/your_data_features.pkl --gpu

# 使用CPU训练
python run_chinese_training.py --data_path ../datapre/your_data_features.pkl

# 自定义参数
python run_chinese_training.py \
    --data_path ../datapre/your_data_features.pkl \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0001 \
    --gpu
```

### 3. 高级训练选项

如果需要更精细的控制，可以直接使用训练脚本：

```bash
python train_chinese_dialoguegcn.py \
    --data_path ../datapre/your_data_features.pkl \
    --n_epochs 50 \
    --batch_size 32 \
    --lr 0.0001 \
    --hidden_dim 200 \
    --dropout 0.5 \
    --valid_split 0.1 \
    --optimizer adam \
    --l2 0.00001 \
    --class_weight \
    --cuda
```

## 📊 模型参数说明

### 数据参数
- `--data_path`: pkl数据文件路径
- `--valid_split`: 验证集比例（默认0.1）
- `--batch_size`: 批处理大小（默认32）
- `--num_workers`: 数据加载器工作进程数（默认0）

### 模型参数
- `--hidden_dim`: 隐藏层维度（默认200）
- `--dropout`: dropout率（默认0.5）

### 训练参数
- `--n_epochs`: 训练轮数（默认50）
- `--optimizer`: 优化器类型
  - `adam`: Adam优化器（推荐）
  - `sgd`: SGD优化器
- `--lr`: 学习率（默认0.0001）
- `--l2`: L2正则化系数（默认0.00001）
- `--momentum`: SGD动量（默认0.0）
- `--class_weight`: 是否使用类别权重

## 🏗️ 模型架构

### SimpleDialogueGCN模型特点

1. **文本编码器**: 双向LSTM处理BERT特征
2. **说话者编码器**: 线性层处理说话者特征
3. **注意力机制**: 多头自注意力融合特征
4. **分类器**: 全连接层进行情绪分类

### 输入特征

- **文本特征**: 768维BERT特征
- **视觉特征**: 100维零向量（占位符）
- **音频特征**: 100维零向量（占位符）
- **说话者特征**: 2维one-hot编码（user=[1,0], robot=[0,1]）

## 📈 训练监控

训练过程中会显示：

- **训练指标**: 损失、准确率、F1分数
- **验证指标**: 损失、准确率、F1分数
- **测试指标**: 损失、准确率、F1分数
- **最佳模型**: 自动保存F1分数最高的模型

## 💾 模型保存

训练过程中会自动保存最佳模型：

- 文件名格式: `chinese_dialoguegcn_model_epoch_X.pth`
- 保存条件: 测试F1分数达到新高时
- 保存内容: 模型状态字典

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批处理大小
   python run_chinese_training.py --data_path your_data.pkl --batch_size 16
   ```

2. **数据加载错误**
   ```bash
   # 检查数据文件格式
   python -c "import pickle; data=pickle.load(open('your_data.pkl', 'rb')); print(data.keys())"
   ```

3. **模型收敛慢**
   ```bash
   # 调整学习率
   python run_chinese_training.py --data_path your_data.pkl --lr 0.001
   ```

### 性能优化

1. **使用GPU**: 确保安装了CUDA和PyTorch GPU版本
2. **调整批处理大小**: 根据GPU内存调整batch_size
3. **调整隐藏层维度**: 根据数据复杂度调整hidden_dim

## 📊 评估结果

训练完成后会显示：

- 最终分类报告
- 混淆矩阵
- 各类别精确率、召回率、F1分数
- 整体准确率和加权F1分数

## 🎉 下一步

训练完成后，您可以：

1. 使用训练好的模型进行推理
2. 在真实对话中测试效果
3. 根据实际效果调整模型参数
4. 考虑集成到实际应用中

## 📝 注意事项

1. **数据质量**: 确保情绪标签分布相对均衡
2. **监控过拟合**: 观察训练和验证指标差异
3. **调整超参数**: 根据实际效果调整学习率、dropout等
4. **保存检查点**: 定期保存模型以防训练中断

## 🔍 数据格式要求

确保您的pkl文件包含以下键：

```python
{
    'video_text': {...},      # 文本特征字典
    'video_speaker': {...},   # 说话者特征字典
    'video_label': {...},     # 情绪标签字典
    'video_audio': {...},     # 音频特征字典
    'video_visual': {...},    # 视觉特征字典
    'video_sentence': {...},  # 句子文本字典
    'trainVids': [...],       # 训练集对话ID列表
    'test_vids': [...]        # 测试集对话ID列表
}
```