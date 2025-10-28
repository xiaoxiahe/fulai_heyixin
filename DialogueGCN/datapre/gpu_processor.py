# GPU加速数据处理器
# 使用GPU大幅提升BERT特征提取速度

import pandas as pd
import numpy as np
import pickle
import torch
import jieba
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import re
from collections import defaultdict, Counter
import random
import os
import gc
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 设置环境变量
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_DISABLE_AUDIO_PROCESSING'] = '1'

class GPUProcessor:
    def __init__(self, model_name='hfl/chinese-roberta-wwm-ext', batch_size=64):
        """
        GPU加速数据处理器
        
        Args:
            model_name: RoBERTa模型名称
            batch_size: 批处理大小（GPU可以处理更大的批次）
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        print(f"正在加载RoBERTa模型: {model_name}")
        print(f"批处理大小: {batch_size}")
        
        # 检查GPU可用性
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device('cpu')
            print("⚠️ 未检测到GPU，将使用CPU模式")
        
        # 加载模型 - 只使用chinese-roberta-wwm-ext，不使用备用模型
        print(f"正在加载模型: {model_name}")
        print("注意: 只使用chinese-roberta-wwm-ext模型，不使用备用模型")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            print(f"✓ 模型加载成功: {type(self.model).__name__}")
            print(f"✓ 模型名称: {model_name}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("请确保:")
            print("1. 网络连接正常")
            print("2. 有足够的磁盘空间")
            print("3. transformers库版本正确")
            print("4. 模型名称正确: hfl/chinese-roberta-wwm-ext")
            raise RuntimeError(f"无法加载chinese-roberta-wwm-ext模型: {e}")
        
        self.model.to(self.device)
        self.model.eval()
        print(f"模型加载完成，使用设备: {self.device}")
        
        # 情绪标签映射（支持多种数据集）
        self.emotion_map = {
            # 映射为neutral (0)
            'neutral': 0,
            'astonished': 0,  # surprise映射为neutral
            'surprised': 0,   # 添加：conversations数据集
            
            # 映射为happy (1)
            'relaxed': 1,
            'happy': 1,
            'grateful': 1,
            'positive-other': 1,
            
            # 映射为sad (2)
            'depress': 2,
            'fear': 2,
            'fearful': 2,     # 添加：conversations数据集
            'negative-other': 2,
            'sadness': 2,
            'sad': 2,         # 添加：conversations数据集
            'worried': 2,
            
            # 映射为angry (3)
            'anger': 3,
            'angry': 3,       # 添加：conversations数据集
            'disgust': 3,
            'disgusted': 3,   # 添加：conversations数据集
        }
        
        # 说话者映射
        self.speaker_map = {}
        self.speaker_counter = 0
        
    def clean_text(self, text):
        """快速清洗文本"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        # 只去除明显的特殊字符，保留更多内容
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def get_speaker_id(self, speaker_name):
        """获取说话者ID"""
        if speaker_name not in self.speaker_map:
            speaker_id = chr(ord('A') + self.speaker_counter)
            self.speaker_map[speaker_name] = speaker_id
            self.speaker_counter += 1
        return self.speaker_map[speaker_name]
    
    def extract_roberta_features_gpu(self, texts):
        """
        GPU加速批量提取RoBERTa特征
        
        Args:
            texts: 文本列表
        
        Returns:
            list: 特征向量列表
        """
        if not texts:
            return []
        
        features = []
        
        # 使用GPU批处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                # 编码
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128  # GPU可以处理更长的序列
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 提取特征
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                features.extend(batch_features)
                
                # 清理GPU内存
                del inputs, outputs, batch_features
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU内存不足，减小批处理大小...")
                    # 减小批处理大小重试
                    smaller_batch_size = self.batch_size // 2
                    for j in range(0, len(batch_texts), smaller_batch_size):
                        small_batch = batch_texts[j:j + smaller_batch_size]
                        try:
                            small_inputs = self.tokenizer(
                                small_batch,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=128
                            )
                            small_inputs = {k: v.to(self.device) for k, v in small_inputs.items()}
                            
                            with torch.no_grad():
                                small_outputs = self.model(**small_inputs)
                                small_features = small_outputs.last_hidden_state[:, 0, :].cpu().numpy()
                            
                            features.extend(small_features)
                            
                            del small_inputs, small_outputs, small_features
                            torch.cuda.empty_cache()
                            
                        except Exception as e2:
                            print(f"小批次处理失败: {e2}")
                            # 如果还是失败，使用零向量填充
                            features.extend([np.zeros(768) for _ in small_batch])
                else:
                    print(f"批处理特征提取失败: {e}")
                    # 回退到单个处理
                    for text in batch_texts:
                        if text and text.strip():
                            try:
                                single_inputs = self.tokenizer(
                                    text,
                                    return_tensors='pt',
                                    padding=True,
                                    truncation=True,
                                    max_length=128
                                )
                                single_inputs = {k: v.to(self.device) for k, v in single_inputs.items()}
                                
                                with torch.no_grad():
                                    single_outputs = self.model(**single_inputs)
                                    single_feature = single_outputs.last_hidden_state[:, 0, :].cpu().numpy()
                                
                                features.append(single_feature[0])
                                
                                del single_inputs, single_outputs, single_feature
                                torch.cuda.empty_cache()
                                
                            except Exception as e2:
                                print(f"单个特征提取失败: {text[:50]}... 错误: {e2}")
                                features.append(np.zeros(768))
                        else:
                            features.append(np.zeros(768))
        
        return features
    
    def load_cped_data(self, file_path):
        """加载CPED数据集"""
        print(f"正在加载CPED数据: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        # 分块读取大文件
        chunk_size = 20000  # GPU可以处理更大的块
        chunks = []
        
        print("正在分块读取数据...")
        for chunk in pd.read_csv(file_path, encoding='utf-8', chunksize=chunk_size):
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        print(f"数据加载完成，共 {len(df)} 条记录")
        print(f"数据列: {list(df.columns)}")
        
        # 检查必要的列
        required_columns = ['TV_ID', 'Utterance', 'Speaker', 'Emotion']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        return df
    
    def group_by_conversation(self, df):
        """按对话ID分组数据"""
        print("正在按对话ID分组数据...")
        conversations = defaultdict(list)
        
        for idx, row in df.iterrows():
            conv_id = row['TV_ID']
            conversations[conv_id].append(row)
        
        print(f"共找到 {len(conversations)} 个对话")
        
        # 显示对话长度分布
        conv_lengths = [len(conv) for conv in conversations.values()]
        print(f"对话长度统计: 平均 {np.mean(conv_lengths):.2f}, 最小 {min(conv_lengths)}, 最大 {max(conv_lengths)}")
        
        return dict(conversations)
    
    def process_single_conversation(self, conv_id, utterances):
        """处理单个对话"""
        # 按索引顺序排序
        utterances = sorted(utterances, key=lambda x: x.name)
        
        # 收集所有文本
        texts = []
        speakers = []
        labels = []
        sentences = []
        
        for utterance in utterances:
            # 清洗文本
            text = self.clean_text(utterance['Utterance'])
            if not text:
                continue
            
            # 获取说话者ID
            speaker_name = utterance['Speaker']
            speaker_id = self.get_speaker_id(speaker_name)
            
            # 获取情绪标签
            emotion = utterance['Emotion'].lower() if pd.notna(utterance['Emotion']) else 'neutral'
            emotion_id = self.emotion_map.get(emotion, 0)
            
            texts.append(text)
            speakers.append(speaker_id)
            labels.append(emotion_id)
            sentences.append(text)
        
        # 检查对话是否有效
        if len(texts) < 2 or len(set(speakers)) < 2:
            return None
        
        # GPU批量提取特征
        print(f"正在GPU提取对话 {conv_id} 的特征...")
        text_features = self.extract_roberta_features_gpu(texts)
        
        return {
            'conv_id': conv_id,
            'speakers': speakers,
            'labels': labels,
            'text_features': text_features,
            'sentences': sentences,
            'speaker_count': len(set(speakers)),
            'turn_count': len(texts)
        }
    
    def convert_to_iemocap_format(self, conversations, output_file):
        """转换为IEMOCAP格式 - 只保留文本特征"""
        print("正在转换为IEMOCAP格式（仅文本特征）...")
        
        # 初始化数据结构 - 只保留文本相关特征
        video_ids = []
        video_speakers = {}
        video_labels = {}
        video_text = {}
        video_sentence = {}
        
        valid_conversations = 0
        
        for conv_id, conv_data in tqdm(conversations.items(), desc="转换对话"):
            if conv_data is None:
                continue
            
            video_ids.append(conv_id)
            video_speakers[conv_id] = conv_data['speakers']
            video_labels[conv_id] = conv_data['labels']
            video_text[conv_id] = conv_data['text_features']
            video_sentence[conv_id] = conv_data['sentences']
            
            valid_conversations += 1
        
        print(f"有效对话数: {valid_conversations}")
        
        # 划分数据集
        random.shuffle(video_ids)
        train_size = int(len(video_ids) * 0.7)
        dev_size = int(len(video_ids) * 0.15)
        
        trainVids = video_ids[:train_size]
        dev_vids = video_ids[train_size:train_size + dev_size]
        test_vids = video_ids[train_size + dev_size:]
        
        # # 增强angry(3)和sad(2)
        # video_ids, video_text, video_labels, video_speakers = self.augment_class(video_ids, video_text, video_labels, video_speakers, target_label=3, ratio=0.2)
        # video_ids, video_text, video_labels, video_speakers = self.augment_class(video_ids, video_text, video_labels, video_speakers, target_label=2, ratio=0.1)


        # 过滤掉有空字段或字段长度不一致的样本
        filtered_ids = []
        for vid in video_ids:
            t = video_text.get(vid, [])
            l = video_labels.get(vid, [])
            s = video_speakers.get(vid, [])
            if t and l and s and len(t) == len(l) == len(s):
                filtered_ids.append(vid)
        video_ids = filtered_ids

        # 类别均衡划分
        from collections import defaultdict
        label_dict = defaultdict(list)
        for vid in video_ids:
            label = video_labels.get(vid, [0])[0]
            label_dict[label].append(vid)
        trainVids, dev_vids, test_vids = [], [], []
        for vids in label_dict.values():
            random.shuffle(vids)
            n = len(vids)
            n_train = int(n * 0.7)
            n_dev = int(n * 0.15)
            trainVids.extend(vids[:n_train])
            dev_vids.extend(vids[n_train:n_train + n_dev])
            test_vids.extend(vids[n_train + n_dev:])
        random.shuffle(trainVids)
        random.shuffle(dev_vids)
        random.shuffle(test_vids)

        print(f"训练集: {len(trainVids)} 个对话")
        print(f"验证集: {len(dev_vids)} 个对话")
        print(f"测试集: {len(test_vids)} 个对话")

        # 统计情绪分布
        self.analyze_emotion_distribution_by_split(video_labels, trainVids, dev_vids, test_vids)

        # 保存数据 - 只保存文本相关特征
        data = (video_ids, video_speakers, video_labels, video_text, trainVids, test_vids)
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"数据保存完成: {output_file}")
        
        # 保存说话者映射
        speaker_map_file = output_file.replace('.pkl', '_speaker_map.pkl')
        with open(speaker_map_file, 'wb') as f:
            pickle.dump(self.speaker_map, f)
        
        print(f"说话者映射保存完成: {speaker_map_file}")
        
        return data
    
    def analyze_data_distribution(self, conversations):
        """分析数据分布"""
        print("\n=== 数据分布分析 ===")
        
        emotion_counts = defaultdict(int)
        speaker_counts = defaultdict(int)
        conversation_lengths = []
        speaker_counts_per_conv = []
        
        for conv_id, conv_data in conversations.items():
            if conv_data is None:
                continue
            
            conversation_lengths.append(conv_data['turn_count'])
            speaker_counts_per_conv.append(conv_data['speaker_count'])
            
            for i, speaker in enumerate(conv_data['speakers']):
                emotion_counts[conv_data['labels'][i]] += 1
                speaker_counts[speaker] += 1
        
        print(f"情绪分布: {dict(emotion_counts)}")
        print(f"说话者分布: {dict(speaker_counts)}")
        print(f"平均对话长度: {np.mean(conversation_lengths):.2f}")
        print(f"平均说话者数量: {np.mean(speaker_counts_per_conv):.2f}")
        print(f"最大对话长度: {max(conversation_lengths)}")
        print(f"最大说话者数量: {max(speaker_counts_per_conv)}")
    
    def analyze_emotion_distribution_by_split(self, video_labels, trainVids, dev_vids, test_vids):
        """分析数据集划分后的情绪分布"""
        print("\n" + "="*60)
        print("数据集划分后情绪分布统计")
        print("="*60)
        
        # 情绪ID到中文名称的映射（与实际映射保持一致）
        emotion_id_to_chinese = {
            0: '中性',      # neutral, astonished
            1: '高兴',      # relaxed, happy, grateful, positive-other
            2: '悲伤',      # depress, fear, negative-other, sadness, worried
            3: '愤怒'       # anger, disgust
        }
        
        # 统计各数据集的情绪分布
        train_emotions = []
        dev_emotions = []
        test_emotions = []
        
        # 训练集情绪统计
        for vid in trainVids:
            if vid in video_labels:
                labels = video_labels[vid]
                train_emotions.extend(labels)
        
        # 验证集情绪统计
        for vid in dev_vids:
            if vid in video_labels:
                labels = video_labels[vid]
                dev_emotions.extend(labels)
        
        # 测试集情绪统计
        for vid in test_vids:
            if vid in video_labels:
                labels = video_labels[vid]
                test_emotions.extend(labels)
        
        # 计算分布
        train_dist = Counter(train_emotions)
        dev_dist = Counter(dev_emotions)
        test_dist = Counter(test_emotions)
        
        # 总分布
        all_emotions = train_emotions + dev_emotions + test_emotions
        total_dist = Counter(all_emotions)
        
        print(f"\n总话语数: {len(all_emotions)}")
        print(f"训练集话语数: {len(train_emotions)}")
        print(f"验证集话语数: {len(dev_emotions)}")
        print(f"测试集话语数: {len(test_emotions)}")
        
        print(f"\n数据集划分比例:")
        print(f"训练集: {len(train_emotions)/len(all_emotions)*100:.1f}%")
        print(f"验证集: {len(dev_emotions)/len(all_emotions)*100:.1f}%")
        print(f"测试集: {len(test_emotions)/len(all_emotions)*100:.1f}%")
        
        print(f"\n情绪分布详情:")
        print(f"{'情绪':<8} {'中文名':<6} {'训练集':<10} {'验证集':<10} {'测试集':<10} {'总计':<8} {'比例':<8}")
        print("-" * 70)
        
        for emotion_id in sorted(total_dist.keys()):
            emotion_chinese = emotion_id_to_chinese.get(emotion_id, f'未知_{emotion_id}')
            
            train_count = train_dist.get(emotion_id, 0)
            dev_count = dev_dist.get(emotion_id, 0)
            test_count = test_dist.get(emotion_id, 0)
            total_count = total_dist.get(emotion_id, 0)
            percentage = total_count / len(all_emotions) * 100
            
            print(f"{emotion_id:<8} {emotion_chinese:<6} {train_count:<10} {dev_count:<10} {test_count:<10} {total_count:<8} {percentage:<7.1f}%")
        
        # 计算每个数据集内部的情绪分布比例
        print(f"\n各数据集内部情绪分布比例:")
        print(f"{'情绪':<8} {'训练集比例':<12} {'验证集比例':<12} {'测试集比例':<12}")
        print("-" * 50)
        
        for emotion_id in sorted(total_dist.keys()):
            emotion_chinese = emotion_id_to_chinese.get(emotion_id, f'未知_{emotion_id}')
            
            train_pct = train_dist.get(emotion_id, 0) / len(train_emotions) * 100 if len(train_emotions) > 0 else 0
            dev_pct = dev_dist.get(emotion_id, 0) / len(dev_emotions) * 100 if len(dev_emotions) > 0 else 0
            test_pct = test_dist.get(emotion_id, 0) / len(test_emotions) * 100 if len(test_emotions) > 0 else 0
            
            print(f"{emotion_id:<8} {train_pct:<11.1f}% {dev_pct:<11.1f}% {test_pct:<11.1f}%")
        
        # 分析数据集平衡性
        print(f"\n=== 数据集平衡性分析 ===")
        
        # 计算每个情绪的比例
        emotion_ratios = {}
        for emotion_id, count in total_dist.items():
            emotion_ratios[emotion_id] = count / len(all_emotions)
        
        # 找出最多和最少的情绪
        max_emotion = max(emotion_ratios.items(), key=lambda x: x[1])
        min_emotion = min(emotion_ratios.items(), key=lambda x: x[1])
        
        print(f"最多情绪: {emotion_id_to_chinese[max_emotion[0]]} ({max_emotion[1]*100:.1f}%)")
        print(f"最少情绪: {emotion_id_to_chinese[min_emotion[0]]} ({min_emotion[1]*100:.1f}%)")
        print(f"不平衡比例: {max_emotion[1]/min_emotion[1]:.2f}:1")
        
        # 计算基尼系数（衡量不平衡程度）
        sorted_ratios = sorted(emotion_ratios.values())
        n = len(sorted_ratios)
        gini = (2 * sum((i + 1) * ratio for i, ratio in enumerate(sorted_ratios))) / (n * sum(sorted_ratios)) - (n + 1) / n
        
        print(f"基尼系数: {gini:.3f} (0=完全平衡, 1=完全不平衡)")
        
        if gini < 0.2:
            print("✅ 数据集相对平衡")
        elif gini < 0.4:
            print("⚠️ 数据集存在一定不平衡")
        else:
            print("❌ 数据集严重不平衡，建议进行数据增强")
        
        # 生成可视化图表
        self.create_emotion_distribution_chart(train_dist, dev_dist, test_dist, emotion_id_to_chinese)
        
        return {
            'train_dist': train_dist,
            'dev_dist': dev_dist, 
            'test_dist': test_dist,
            'total_dist': total_dist,
            'train_utterances': len(train_emotions),
            'dev_utterances': len(dev_emotions),
            'test_utterances': len(test_emotions),
            'total_utterances': len(all_emotions)
        }
    
    def create_emotion_distribution_chart(self, train_dist, dev_dist, test_dist, emotion_id_to_chinese):
        """创建情绪分布可视化图表"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 准备数据
            emotions = sorted(set(train_dist.keys()) | set(dev_dist.keys()) | set(test_dist.keys()))
            emotion_names = [emotion_id_to_chinese.get(e, f'未知_{e}') for e in emotions]
            
            train_counts = [train_dist.get(e, 0) for e in emotions]
            dev_counts = [dev_dist.get(e, 0) for e in emotions]
            test_counts = [test_dist.get(e, 0) for e in emotions]
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('情绪分布分析', fontsize=16, fontweight='bold')
            
            # 1. 总体分布饼图
            ax1 = axes[0, 0]
            total_counts = [train_dist.get(e, 0) + dev_dist.get(e, 0) + test_dist.get(e, 0) for e in emotions]
            colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
            
            wedges, texts, autotexts = ax1.pie(total_counts, labels=emotion_names, autopct='%1.1f%%', 
                                              colors=colors, startangle=90)
            ax1.set_title('总体情绪分布')
            
            # 2. 各数据集分布对比柱状图
            ax2 = axes[0, 1]
            x = np.arange(len(emotions))
            width = 0.25
            
            ax2.bar(x - width, train_counts, width, label='训练集', alpha=0.8)
            ax2.bar(x, dev_counts, width, label='验证集', alpha=0.8)
            ax2.bar(x + width, test_counts, width, label='测试集', alpha=0.8)
            
            ax2.set_xlabel('情绪类别')
            ax2.set_ylabel('话语数量')
            ax2.set_title('各数据集情绪分布对比')
            ax2.set_xticks(x)
            ax2.set_xticklabels(emotion_names, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 训练集分布
            ax3 = axes[1, 0]
            ax3.bar(emotion_names, train_counts, color='skyblue', alpha=0.7)
            ax3.set_title('训练集情绪分布')
            ax3.set_ylabel('话语数量')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # 4. 测试集分布
            ax4 = axes[1, 1]
            ax4.bar(emotion_names, test_counts, color='lightcoral', alpha=0.7)
            ax4.set_title('测试集情绪分布')
            ax4.set_ylabel('话语数量')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            output_file = 'emotion_distribution_analysis.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n📊 情绪分布图表已保存: {output_file}")
            
            # 显示图表
            plt.show()
            
        except ImportError:
            print("⚠️ matplotlib未安装，跳过图表生成")
        except Exception as e:
            print(f"⚠️ 图表生成失败: {e}")
    
    def process_cped_data(self, input_file, output_file):
        """处理CPED数据的主函数"""
        print("开始GPU加速处理CPED数据...")
        print("注意: 使用GPU加速，处理速度大幅提升")
        
        # 1. 加载数据
        df = self.load_cped_data(input_file)
        
        # 2. 按对话分组
        conversations = self.group_by_conversation(df)
        
        # 3. 处理每个对话
        processed_conversations = {}
        for conv_id, utterances in tqdm(conversations.items(), desc="处理对话"):
            processed_conv = self.process_single_conversation(conv_id, utterances)
            if processed_conv:
                processed_conversations[conv_id] = processed_conv
        
        # 4. 分析数据分布
        self.analyze_data_distribution(processed_conversations)
        
        # 5. 转换为IEMOCAP格式
        data = self.convert_to_iemocap_format(processed_conversations, output_file)
        
        print("CPED数据处理完成！")
        return data

    def augment_class(self, video_ids, video_text_dict, video_label_dict, video_speakers_dict, target_label, ratio=0.5):
        """
        对指定类别做简单数据增强：复制+顺序打乱
        target_label: 目标类别（如0=neutral, 2=sad, 3=angry）
        ratio: 增强比例
        """
        import random
        from copy import deepcopy
        new_vids = []
        for vid in video_ids:
            labels = video_label_dict.get(vid, [])
            speakers = video_speakers_dict.get(vid, [])
            # 只增强长度大于0且字段长度一致的样本
            if not labels or not speakers:
                continue
            if not (len(labels) == len(speakers) == len(video_text_dict.get(vid, []))):
                continue
            if all(l == target_label for l in labels):
                if random.random() < ratio:
                    new_vid = vid + f'_aug{target_label}'
                    new_text = deepcopy(video_text_dict[vid])
                    new_labels = [target_label for _ in new_text]
                    new_speakers = deepcopy(video_speakers_dict[vid])
                    # 保证打乱后三者长度一致
                    zipped = list(zip(new_text, new_labels, new_speakers))
                    random.shuffle(zipped)
                    new_text, new_labels, new_speakers = zip(*zipped)
                    video_text_dict[new_vid] = list(new_text)
                    video_label_dict[new_vid] = list(new_labels)
                    video_speakers_dict[new_vid] = list(new_speakers)
                    new_vids.append(new_vid)
        video_ids.extend(new_vids)
        return video_ids, video_text_dict, video_label_dict, video_speakers_dict
    
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU加速CPED数据处理器')
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出PKL文件路径')
    parser.add_argument('--model', type=str, default='hfl/chinese-roberta-wwm-ext', help='RoBERTa模型名称')
    parser.add_argument('--batch_size', type=int, default=64, help='批处理大小')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 初始化处理器
    processor = GPUProcessor(model_name=args.model, batch_size=args.batch_size)
    
    # 处理数据
    data = processor.process_cped_data(args.input, args.output)
    
    print(f"数据处理完成！输出文件: {args.output}")

if __name__ == "__main__":
    main()
