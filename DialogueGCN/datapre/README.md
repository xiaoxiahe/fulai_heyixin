# 数据预处理模块

这个模块用于将各种格式的中文多轮对话情绪识别数据集转换为DialogueGCN可训练的格式。

## 文件说明

- `cped_data_processor.py`: CPED数据集专用处理器
- `validate_cped_data.py`: 数据验证脚本
- `data_converter.py`: 通用数据格式转换工具
- `requirements.txt`: 依赖包列表

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. GPU加速处理（推荐，速度最快）

```bash
# 使用GPU加速处理器
python gpu_start.py test_split.csv

# 指定输出文件名
python gpu_start.py test_split.csv my_features.pkl

```

### 3. CPU模式处理（无需GPU）

```bash
# 使用CPU优化的处理器
python run_cpu_preprocessing.py --input test_split.csv --output cped_features.pkl

# 估算处理时间
python run_cpu_preprocessing.py --input test_split.csv --output cped_features.pkl --estimate_time
```

### 4. 处理CPED数据集（原版）

```bash
python cped_data_processor.py --input cped_data.csv --output cped_features.pkl
```

### 4. 验证数据

```bash
python validate_cped_data.py --data cped_features.pkl --speaker_map cped_features_speaker_map.pkl
```

### 5. 转换其他格式数据

#### JSON格式
```bash
python data_converter.py --input data.json --output features.pkl --format json
```

#### CSV格式
```bash
python data_converter.py --input data.csv --output features.pkl --format csv \
    --conv_id_col TV_ID \
    --text_col Dialogue_Utterance \
    --speaker_col Speaker \
    --emotion_col Emotion
```

## 数据格式要求

### 输入数据格式

#### CPED数据集
- CSV文件，包含以下列：
  - `TV_ID`: 对话ID
  - `Dialogue_Utterance`: 对话文本
  - `Speaker`: 说话者
  - `Emotion`: 情绪标签

#### JSON格式
```json
{
    "conv_001": {
        "utterances": [
            {
                "speaker": "A",
                "text": "今天天气真不错",
                "emotion": "happy"
            },
            {
                "speaker": "B",
                "text": "是啊，适合出去走走",
                "emotion": "happy"
            }
        ]
    }
}
```

### 输出数据格式

处理后的数据将保存为PKL文件，包含以下内容：
- `video_ids`: 对话ID列表
- `video_speakers`: 说话者信息字典
- `video_labels`: 情绪标签字典
- `video_text`: 文本特征字典（768维BERT特征）
- `video_audio`: 音频特征字典（100维零向量）
- `video_visual`: 视觉特征字典（100维零向量）
- `video_sentence`: 句子文本字典
- `trainVids`: 训练集对话ID列表
- `test_vids`: 测试集对话ID列表

## 情绪标签映射（CPED数据集13种情绪）

### 映射规则：
- **happy (1)**: `relaxed`, `happy`, `grateful`, `positive-other`
- **neutral (0)**: `neutral`
- **surprise (5)**: `astonished`
- **sad (2)**: `depress`, `fear`, `negative-other`, `sadness`, `worried`
- **angry (3)**: `anger`, `disgust`

### 原始13种情绪标签：
1. `relaxed` → happy
2. `happy` → happy
3. `grateful` → happy
4. `positive-other` → happy
5. `neutral` → neutral
6. `astonished` → surprise
7. `depress` → sad
8. `fear` → sad
9. `negative-other` → sad
10. `sadness` → sad
11. `worried` → sad
12. `anger` → angry
13. `disgust` → angry

## 说话者映射

说话者将按出现顺序映射为字母：
- 第一个说话者: A
- 第二个说话者: B
- 第三个说话者: C
- 以此类推...

## 注意事项

1. 确保输入数据包含至少2轮对话和2个不同说话者
2. 文本将自动清洗，去除特殊字符
3. 使用BERT提取768维特征向量
4. 数据将按7:1.5:1.5的比例划分为训练集、验证集和测试集
5. 说话者映射将保存为单独的文件

## CPU模式说明

### 优势
- ✅ 无需GPU，普通电脑即可运行
- ✅ 内存占用相对较少
- ✅ 兼容性好，适合各种环境

### 劣势
- ⚠️ 处理速度较慢（比GPU慢5-10倍）
- ⚠️ 大数据集处理时间较长

### 性能优化建议
1. **调整批处理大小**：根据内存大小调整（建议4-16）
2. **分批处理**：如果数据量很大，可以分批处理
3. **关闭其他程序**：释放内存资源
4. **使用SSD**：提高磁盘读写速度

## 错误处理

- 如果文本为空或无效，将使用零向量填充
- 如果情绪标签无效，将映射为中性(0)
- 如果对话无效（少于2轮或少于2个说话者），将被跳过

## 性能优化

- 使用GPU加速BERT特征提取
- 批量处理提高效率
- 进度条显示处理状态
