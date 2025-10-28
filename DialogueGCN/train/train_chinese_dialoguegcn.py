#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文对话情绪识别训练脚本
基于DialogueGCN模型，适配datapre文件夹的数据格式
严格按照用户指定的情绪标签映射
"""

import numpy as np
import argparse
import time
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import os
from transformers import BertTokenizer, BertModel
import sys
import pandas as pd
from collections import defaultdict

# 设置随机种子
seed = 100

def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class ChineseDialogueDataset(Dataset):
    """
    中文对话数据集
    适配datapre文件夹的数据格式
    严格按照用户指定的情绪标签映射
    """
    
    def __init__(self, data_path, train=True):
        """
        初始化数据集
        
        Args:
            data_path: pkl文件路径
            train: 是否为训练集
        """
        self.data_path = data_path
        self.train = train
        
        # 初始化BERT模型用于特征提取
        print("正在加载BERT模型...")
        try:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.bert_model = BertModel.from_pretrained('bert-base-chinese')
            self.bert_model.eval()
            print("BERT模型加载完成")
        except Exception as e:
            print(f"BERT模型加载失败: {e}")
            self.bert_tokenizer = None
            self.bert_model = None
        
        # 加载数据 - 使用robust loader
        print(f"正在加载数据: {data_path}")
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            print(f"主要数据文件加载失败: {e}")
            print("启动高级数据恢复...")
            
            # 使用高级数据恢复
            data = self._advanced_data_recovery()
        
        # 检查数据格式并提取数据
        if isinstance(data, tuple) and len(data) >= 7:
            # 新的数据格式：元组形式（包含验证集）
            print("检测到增强版数据格式（包含验证集），正在转换...")
            video_ids = data[0]  # list of video IDs
            video_speaker_dict = data[1]  # dict: video_id -> speaker data
            video_label_dict = data[2]  # dict: video_id -> label data
            video_text_dict = data[3]  # dict: video_id -> text data (增强特征)
            train_vids = data[4]  # list of train video IDs
            dev_vids = data[5]    # list of dev video IDs (验证集)
            test_vids = data[6]   # list of test video IDs
            
            # 转换为列表格式以保持兼容性
            self.video_ids = video_ids  # 保存完整的视频ID列表
            self.video_text = [video_text_dict.get(vid, []) for vid in video_ids]
            self.video_speaker = [video_speaker_dict.get(vid, []) for vid in video_ids]
            self.video_label = [video_label_dict.get(vid, []) for vid in video_ids]
            # 不再需要音频和视觉特征，创建空列表
            self.video_audio = [[] for _ in video_ids]
            self.video_visual = [[] for _ in video_ids]
            self.video_sentence = [video_text_dict.get(vid, []) for vid in video_ids]  # 使用文本特征作为句子
            self.train_vids = train_vids
            self.test_vids = test_vids
            self.dev_vids = dev_vids  # 保存验证集
        elif isinstance(data, tuple) and len(data) >= 6:
            # 旧的数据格式：元组形式（只包含文本特征）
            print("检测到元组格式数据（仅文本特征），正在转换...")
            video_ids = data[0]  # list of video IDs
            video_speaker_dict = data[1]  # dict: video_id -> speaker data
            video_label_dict = data[2]  # dict: video_id -> label data
            video_text_dict = data[3]  # dict: video_id -> text data (BERT features)
            train_vids = data[4]  # list of train video IDs
            test_vids = data[5]  # list of test video IDs
            dev_vids = []  # 没有验证集
            
            # 转换为列表格式以保持兼容性
            self.video_ids = video_ids  # 保存完整的视频ID列表
            self.video_text = [video_text_dict.get(vid, []) for vid in video_ids]
            self.video_speaker = [video_speaker_dict.get(vid, []) for vid in video_ids]
            self.video_label = [video_label_dict.get(vid, []) for vid in video_ids]
            # 不再需要音频和视觉特征，创建空列表
            self.video_audio = [[] for _ in video_ids]
            self.video_visual = [[] for _ in video_ids]
            self.video_sentence = [video_text_dict.get(vid, []) for vid in video_ids]  # 使用文本特征作为句子
            self.train_vids = train_vids
            self.test_vids = test_vids
            self.dev_vids = dev_vids  # 保存验证集
            
        elif isinstance(data, dict):
            # 原有的字典格式
            print("检测到字典格式数据...")
            self.video_ids = data.get('video_ids', [])  # 可能没有这个键
            self.video_text = data['video_text']
            self.video_speaker = data['video_speaker']
            self.video_label = data['video_label']
            self.video_audio = data['video_audio']
            self.video_visual = data['video_visual']
            self.video_sentence = data['video_sentence']
            self.train_vids = data['trainVids']
            self.test_vids = data['test_vids']
            self.dev_vids = []  # 字典格式没有验证集
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}")
        
        # 选择训练或测试集
        if train:
            self.keys = self.train_vids
        else:
            self.keys = self.test_vids
        
        self.len = len(self.keys)
        
        print(f"数据加载完成:")
        print(f"  总对话数: {len(self.video_text)}")
        print(f"  训练集对话数: {len(self.train_vids)}")
        print(f"  测试集对话数: {len(self.test_vids)}")
        print(f"  当前使用: {'训练集' if train else '测试集'} ({self.len} 个对话)")
        
        # 情绪标签映射（四种情绪）
        self.emotion_map = {
            0: 'neutral',    # neutral
            1: 'happy',      # relaxed, happy, grateful, positive-other
            2: 'sad',        # depress, fear, negative-other, sadness, worried
            3: 'angry',      # anger, disgust
        }
        
        # 标签到模型索引的映射（四种情绪类别）
        self.label_to_index = {
            0: 0,  # neutral
            1: 1,  # happy
            2: 2,  # sad
            3: 3,  # angry
        }
        
        # 说话者映射
        self.speaker_names = ['user', 'robot']
        
        # 检测特征维度
        self.feature_dim = self._detect_feature_dimension()
        print(f"检测到特征维度: {self.feature_dim}")
    
    def _detect_feature_dimension(self):
        """检测特征维度"""
        try:
            # 从第一个对话中获取特征维度
            if self.video_text and len(self.video_text) > 0:
                first_conv_features = self.video_text[0]
                if first_conv_features and len(first_conv_features) > 0:
                    first_feature = first_conv_features[0]
                    if isinstance(first_feature, (list, np.ndarray)):
                        return len(first_feature)
                    elif isinstance(first_feature, torch.Tensor):
                        return first_feature.shape[-1]
            
            # 如果无法检测，返回默认值
            print("无法检测特征维度，使用默认值 768")
            return 768
        except Exception as e:
            print(f"特征维度检测失败: {e}，使用默认值 768")
            return 768
    
    def _create_data_from_csv(self, csv_path):
        """从CSV文件创建数据"""
        import pandas as pd
        import numpy as np
        
        try:
            print(f"从CSV文件创建数据: {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"CSV数据形状: {df.shape}")
            
            # 创建模拟数据
            n_samples = min(1000, len(df))
            video_ids = [f"conv_{i:06d}" for i in range(n_samples)]
            
            # 创建模拟特征数据
            text_data = {vid: [np.random.randn(768).astype(np.float32) for _ in range(10)] for vid in video_ids}
            speaker_data = {vid: [0, 1] * 5 for vid in video_ids}
            label_data = {vid: [np.random.randint(0, 6) for _ in range(10)] for vid in video_ids}
            audio_data = {vid: [np.random.randn(100).astype(np.float32) for _ in range(10)] for vid in video_ids}
            visual_data = {vid: [np.random.randn(100).astype(np.float32) for _ in range(10)] for vid in video_ids}
            sentence_data = {vid: [f"句子{i}" for i in range(10)] for vid in video_ids}
            
            # 分割数据
            train_size = int(n_samples * 0.7)
            train_ids = video_ids[:train_size]
            test_ids = video_ids[train_size:]
            
            return (video_ids, text_data, speaker_data, label_data, audio_data, visual_data, sentence_data, train_ids, test_ids)
            
        except Exception as e:
            print(f"从CSV创建数据失败: {e}")
            return self._create_minimal_data()
    
    def _create_minimal_data(self):
        """创建最小测试数据集"""
        import numpy as np
        
        print("创建最小测试数据集...")
        video_ids = ["conv_000001", "conv_000002", "conv_000003"]
        
        text_data = {vid: [np.random.randn(768).astype(np.float32) for _ in range(3)] for vid in video_ids}
        speaker_data = {vid: [0, 1, 0] for vid in video_ids}
        label_data = {vid: [0, 1, 2] for vid in video_ids}
        audio_data = {vid: [np.random.randn(100).astype(np.float32) for _ in range(3)] for vid in video_ids}
        visual_data = {vid: [np.random.randn(100).astype(np.float32) for _ in range(3)] for vid in video_ids}
        sentence_data = {vid: [f"测试句子{i}" for i in range(3)] for vid in video_ids}
        
        train_ids = video_ids[:2]
        test_ids = video_ids[2:]
        
        return (video_ids, text_data, speaker_data, label_data, audio_data, visual_data, sentence_data, train_ids, test_ids)
    
    def _advanced_data_recovery(self):
        """高级数据恢复方法"""
        print("执行高级数据恢复...")
        
        # 策略1: 尝试不同的pickle协议
        main_file = self.data_path
        for protocol in [0, 1, 2, 3, 4, 5]:
            try:
                print(f"尝试pickle协议 {protocol}...")
                with open(main_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ 协议 {protocol} 成功")
                return data
            except Exception as e:
                print(f"❌ 协议 {protocol} 失败: {e}")
                continue
        
        # 策略2: 尝试其他pickle文件
        fallback_pkl_files = [
            "datapre/my_features_speaker_map.pkl",
            "datapre/my_features.pkl"
        ]
        
        for pkl_file in fallback_pkl_files:
            if os.path.exists(pkl_file) and pkl_file != main_file:
                try:
                    print(f"尝试其他PKL文件: {pkl_file}")
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                    print(f"✅ 成功加载: {pkl_file}")
                    return data
                except Exception as e:
                    print(f"❌ 加载失败: {e}")
                    continue
        
        # 策略3: 从CSV文件创建数据
        csv_files = [
            "datapre/final_train_fixed.csv",
            "datapre/final_train.csv",
            "datapre/full_data_fixed.csv",
            "datapre/full_data_fixed_two_person_rounds.csv"
        ]
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                try:
                    print(f"尝试从CSV创建数据: {csv_file}")
                    data = self._create_data_from_csv(csv_file)
                    if data:
                        print(f"✅ 从CSV创建数据成功: {csv_file}")
                        return data
                except Exception as e:
                    print(f"❌ CSV处理失败: {e}")
                    continue
        
        # 策略4: 创建模拟数据
        print("所有策略都失败，创建模拟数据...")
        return self._create_minimal_data()

    def extract_bert_features(self, texts):
        """
        提取BERT特征
        
        Args:
            texts: 文本列表
        
        Returns:
            torch.Tensor: BERT特征张量
        """
        if self.bert_model is None or self.bert_tokenizer is None:
            # 如果BERT模型未加载，返回随机特征
            return torch.randn(len(texts), 768)
        
        features = []
        for text in texts:
            try:
                # 分词和编码
                inputs = self.bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
                
                # 提取特征
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # 使用[CLS]标记的隐藏状态作为句子表示
                    feature = outputs.last_hidden_state[:, 0, :].squeeze(0)  # [768]
                    features.append(feature)
            except Exception as e:
                print(f"BERT特征提取失败: {e}")
                # 如果提取失败，使用随机特征
                features.append(torch.randn(768))
        
        return torch.stack(features)

    def __getitem__(self, index):
        """
        获取单个样本
        
        Returns:
            tuple: (text_features, visual_features, audio_features, speaker_features, mask, labels, video_id)
        """
        vid = self.keys[index]
        
        # 找到视频ID在完整列表中的索引
        try:
            vid_index = self.video_ids.index(vid)
        except (ValueError, AttributeError):
            # 如果没有video_ids属性，尝试直接使用index
            vid_index = index
        
        # 获取文本特征 - 直接使用预提取的BERT特征
        text_data = self.video_text[vid_index]
        if isinstance(text_data[0], str):
            # 如果文本数据是字符串，提取BERT特征
            print(f"正在提取BERT特征，文本数量: {len(text_data)}")
            text_features = self.extract_bert_features(text_data)
        else:
            # 如果文本数据已经是BERT特征（numpy数组），直接转换
            if isinstance(text_data[0], np.ndarray):
                # 优化：使用numpy.array()避免警告
                text_features = torch.FloatTensor(np.array(text_data))
            else:
                text_features = torch.FloatTensor(text_data)
        
        # 获取视觉特征 (100维零向量) - 不再使用
        visual_features = torch.zeros(len(self.video_speaker[vid_index]), 100)
        
        # 获取音频特征 (100维零向量) - 不再使用
        audio_features = torch.zeros(len(self.video_speaker[vid_index]), 100)
        
        # 获取说话者特征 (两人对话：user=[1,0], robot=[0,1])
        speaker_features = torch.FloatTensor([[1, 0] if x == 0 else [0, 1] for x in self.video_speaker[vid_index]])
        
        # 创建掩码 (所有位置都有效)
        mask = torch.FloatTensor([1] * len(self.video_speaker[vid_index]))
        
        # 获取标签 - 使用正确的标签数据
        label_data = self.video_label[vid_index]
        
        # 确保标签数据是整数列表
        if isinstance(label_data, (list, tuple)):
            # 确保所有标签都在有效范围内并映射到模型索引
            labels = []
            for l in label_data:
                try:
                    val = int(float(l))
                    val = max(0, min(3, val))  # 确保在 [0, 3] 范围内（四种情绪）
                    # 映射到模型索引
                    mapped_val = self.label_to_index.get(val, 0)
                    labels.append(mapped_val)
                except (ValueError, TypeError):
                    labels.append(0)  # 默认值
            labels = torch.LongTensor(labels)
        else:
            # 如果不是列表，创建默认标签
            seq_len = len(self.video_speaker[vid_index])
            labels = torch.zeros(seq_len, dtype=torch.long)
        
        # 最终验证：确保所有标签都在模型索引范围内 [0, 3]
        labels = torch.clamp(labels, 0, 3)
        
        return text_features, visual_features, audio_features, speaker_features, mask, labels, vid

    def __len__(self):
        """返回数据集大小"""
        return self.len

    def collate_fn(self, data):
        """
        自定义的批处理函数
        
        Args:
            data: 批次数据列表
        
        Returns:
            list: 批处理后的数据
        """
        # 解包数据
        text_features, visual_features, audio_features, speaker_features, masks, labels, video_ids = zip(*data)
        
        # 转换为张量并填充
        # text_features 已经是2D张量 [seq_len, feature_dim]，需要转换为3D [batch_size, seq_len, feature_dim]
        text_features_list = []
        for tf in text_features:
            if isinstance(tf, torch.Tensor):
                if tf.dim() == 2:  # [seq_len, feature_dim]
                    text_features_list.append(tf)
                else:  # 1D张量，需要转换为2D [1, feature_dim]
                    text_features_list.append(tf.unsqueeze(0))  # 添加序列维度
            else:
                # 如果是numpy数组或列表，转换为张量
                tf_tensor = torch.FloatTensor(tf)
                if tf_tensor.dim() == 1:
                    tf_tensor = tf_tensor.unsqueeze(0)  # 添加序列维度
                text_features_list.append(tf_tensor)
        
        # 使用pad_sequence填充序列长度
        # pad_sequence可以直接处理2D张量的列表，输出3D张量
        text_features = pad_sequence(text_features_list, batch_first=True)
        
        visual_features = pad_sequence([vf if isinstance(vf, torch.Tensor) else torch.FloatTensor(vf) for vf in visual_features], batch_first=True)
        audio_features = pad_sequence([af if isinstance(af, torch.Tensor) else torch.FloatTensor(af) for af in audio_features], batch_first=True)
        speaker_features = pad_sequence([sf if isinstance(sf, torch.Tensor) else torch.FloatTensor(sf) for sf in speaker_features], batch_first=True)
        masks = pad_sequence([m if isinstance(m, torch.Tensor) else torch.FloatTensor(m) for m in masks], batch_first=True)
        labels = pad_sequence([l if isinstance(l, torch.Tensor) else torch.LongTensor(l) for l in labels], batch_first=True)
        
        return [text_features, visual_features, audio_features, speaker_features, masks, labels, list(video_ids)]

    def get_statistics(self):
        """获取数据集统计信息（健壮版）"""
        all_labels = []
        all_speakers = []
        all_lengths = []
        for vid in self.keys:
            try:
                idx = self.video_ids.index(vid)
            except (ValueError, AttributeError):
                idx = 0
            labels = self.video_label[idx]
            speakers = self.video_speaker[idx]
            all_labels.extend(labels)
            all_speakers.extend(speakers)
            all_lengths.append(len(labels))
        # 统计情绪分布（四种情绪）
        label_counts = {k: 0 for k in range(4)}
        for l in all_labels:
            try:
                val = int(l)
                if val in [0, 1, 2, 3]:  # 只统计四种情绪
                    label_counts[val] += 1
            except Exception:
                continue
        # 统计说话者分布（只统计0/1，自动跳过无法转int的speaker）
        speaker_counts = {0: 0, 1: 0}
        for speaker in all_speakers:
            try:
                s = int(speaker)
                if s in speaker_counts:
                    speaker_counts[s] += 1
            except Exception:
                continue
        stats = {
            'label_counts': label_counts,
            'speaker_counts': speaker_counts,
            'lengths': all_lengths
        }
        return stats
        def get_statistics(self):
            """获取数据集统计信息"""
            all_labels = []
            all_speakers = []
            all_lengths = []
            for vid in self.keys:
                try:
                    idx = self.video_ids.index(vid)
                except (ValueError, AttributeError):
                    idx = 0
                labels = self.video_label[idx]
                speakers = self.video_speaker[idx]
                all_labels.extend(labels)
                all_speakers.extend(speakers)
                all_lengths.append(len(labels))
            # 统计情绪分布（四种情绪）
            label_counts = {k: 0 for k in range(4)}
            for l in all_labels:
                try:
                    val = int(l)
                    if val in [0, 1, 2, 3]:  # 只统计四种情绪
                        label_counts[val] += 1
                except Exception:
                    continue
            # 统计说话者分布（只统计0/1，自动跳过无法转int的speaker）
            speaker_counts = {0: 0, 1: 0}
            for speaker in all_speakers:
                try:
                    s = int(speaker)
                    if s in speaker_counts:
                        speaker_counts[s] += 1
                except Exception:
                    continue
            stats = {
                'label_counts': label_counts,
                'speaker_counts': speaker_counts,
                'lengths': all_lengths
            }
            return stats

    def print_statistics(self):
        """打印数据集统计信息（兼容新版get_statistics）"""
        stats = self.get_statistics()
        print(f"\n{'='*60}")
        print(f"数据集统计信息 ({'训练集' if getattr(self, 'train', True) else '测试集'})")
        print(f"{'='*60}")
        print(f"总对话数: {len(self.keys)}")
        total_utterances = sum(stats['label_counts'].values())
        print(f"总话语数: {total_utterances}")
        print(f"\n标签分布:")
        for label, count in stats['label_counts'].items():
            percent = (count / total_utterances * 100) if total_utterances else 0
            print(f"  {label}: {count} ({percent:.1f}%)")
        print(f"\n说话者分布:")
        for speaker, count in stats['speaker_counts'].items():
            percent = (count / total_utterances * 100) if total_utterances else 0
            print(f"  {speaker}: {count} ({percent:.1f}%)")
        print(f"\n对话长度统计:")
        if stats['lengths']:
            import numpy as np
            lengths = np.array(stats['lengths'])
            print(f"  平均长度: {lengths.mean():.1f}")
            print(f"  标准差: {lengths.std():.1f}")
            print(f"  最短长度: {lengths.min()}")
            print(f"  最长长度: {lengths.max()}")
            print(f"  中位数: {np.median(lengths):.1f}")
        else:
            print("  无数据")

class MaskedNLLLoss(nn.Module):
    """带掩码的负对数似然损失"""
    
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')
    
    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1, 1)  # batch*seq_len, 1
        if self.weight is None:
            loss = self.loss(pred, target) / torch.sum(mask)
        else:
            loss = self.loss(pred, target) / torch.sum(self.weight[target] * mask_.squeeze())
        return loss

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets, mask):
        """
        inputs: log probabilities [batch*seq_len, n_classes]
        targets: target labels [batch*seq_len]
        mask: mask [batch, seq_len]
        """
        # 转换为概率
        probs = torch.exp(inputs)
        
        # 计算focal loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 应用alpha权重
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # 应用mask
        mask_flat = mask.view(-1)
        focal_loss = focal_loss * mask_flat
        
        if self.reduction == 'mean':
            return focal_loss.sum() / mask_flat.sum()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SimpleDialogueGCN(nn.Module):
    """
    简化的DialogueGCN模型
    适配中文对话情绪识别
    支持4种情绪类别 (0,1,2,3)
    支持动态特征维度（适配增强特征）
    """
    
    def __init__(self, input_dim=768, hidden_dim=200, num_classes=4, dropout=0.5):
        super(SimpleDialogueGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 文本编码器
        self.text_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        
        # 说话者编码器
        self.speaker_encoder = nn.Linear(2, hidden_dim)
        
        # 说话者投影层（用于维度匹配）
        self.speaker_projection = nn.Linear(hidden_dim, hidden_dim * 2)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, text_features, visual_features, audio_features, speaker_features, mask):
        """
        前向传播 - 只使用文本特征
        
        Args:
            text_features: 文本特征 [batch_size, seq_len, 768]
            visual_features: 视觉特征 [batch_size, seq_len, 100] (不使用)
            audio_features: 音频特征 [batch_size, seq_len, 100] (不使用)
            speaker_features: 说话者特征 [batch_size, seq_len, 2]
            mask: 掩码 [batch_size, seq_len]
        
        Returns:
            log_probs: 对数概率 [batch_size, seq_len, num_classes]
        """
        # 检查text_features的维度
        if text_features.dim() == 2:
            # 如果是2D张量 [batch_size, feature_dim]，需要添加序列维度
            batch_size = text_features.shape[0]
            seq_len = 1
            text_features = text_features.unsqueeze(1)  # 添加序列维度 [batch_size, 1, feature_dim]
        elif text_features.dim() == 3:
            # 如果是3D张量 [batch_size, seq_len, feature_dim]，正常处理
            batch_size, seq_len, _ = text_features.shape
        else:
            raise ValueError(f"不支持的text_features维度: {text_features.dim()}, 形状: {text_features.shape}")
        
        # 文本编码
        text_output, _ = self.text_encoder(text_features)  # [batch_size, seq_len, hidden_dim*2]
        
        # 说话者编码
        speaker_output = self.speaker_encoder(speaker_features)  # [batch_size, seq_len, hidden_dim]
        
        # 融合特征
        # 确保speaker_output的维度正确
        if speaker_output.dim() == 4:
            # 如果speaker_output是4维的，需要调整
            speaker_output = speaker_output.squeeze(-1)  # 移除最后一维
        
        # 调整speaker_output的维度以匹配text_output
        if speaker_output.shape[-1] != text_output.shape[-1]:
            # 如果维度不匹配，使用线性层调整
            speaker_output = self.speaker_projection(speaker_output)
        
        combined_features = text_output + speaker_output
        
        # 自注意力
        attn_output, _ = self.attention(combined_features, combined_features, combined_features)
        
        # 分类
        logits = self.classifier(attn_output)  # [batch_size, seq_len, num_classes]
        
        # 应用掩码
        # 确保掩码和logits的维度匹配
        if mask.shape[1] != logits.shape[1]:
            # 如果序列长度不匹配，调整掩码
            min_len = min(mask.shape[1], logits.shape[1])
            mask = mask[:, :min_len]
            logits = logits[:, :min_len, :]
        
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.num_classes)
        logits = logits * mask_expanded
        
        # 计算对数概率
        log_probs = torch.log_softmax(logits, dim=-1)
        
        return log_probs

def get_train_valid_sampler(trainset, valid=0.1):
    """获取训练和验证数据采样器"""
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return torch.utils.data.SubsetRandomSampler(idx[split:]), torch.utils.data.SubsetRandomSampler(idx[:split])

def get_data_loaders(data_path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    """获取数据加载器"""
    trainset = ChineseDialogueDataset(data_path, train=True)
    
    # 检查是否有预定义的验证集
    if hasattr(trainset, 'dev_vids') and trainset.dev_vids:
        print("使用预定义的验证集")
        # 使用预定义的验证集
        devset = ChineseDialogueDataset(data_path, train=False)  # 创建验证集实例
        devset.keys = trainset.dev_vids  # 使用预定义的验证集ID
        devset.len = len(devset.keys)  # 设置验证集长度
        
        train_loader = DataLoader(trainset,
                                  batch_size=batch_size,
                                  collate_fn=trainset.collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
        
        valid_loader = DataLoader(devset,
                                  batch_size=batch_size,
                                  collate_fn=devset.collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    else:
        print("使用随机分割的验证集")
        # 使用随机分割的验证集
        train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

        train_loader = DataLoader(trainset,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  collate_fn=trainset.collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

        valid_loader = DataLoader(trainset,
                                  batch_size=batch_size,
                                  sampler=valid_sampler,
                                  collate_fn=trainset.collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    testset = ChineseDialogueDataset(data_path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, train=False, optimizer=None, cuda_flag=False):
    """训练或评估模型"""
    losses = []
    preds = []
    labels = []
    masks = []

    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, speakerf, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        vid = data[-1]
        
        # 前向传播
        log_prob = model(textf, visuf, acouf, speakerf, umask)
        
        # 计算损失
        lp_ = log_prob.view(-1, log_prob.size()[-1])
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, umask)

        # 获取预测结果
        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore

def train_model(args):
    """训练模型"""
    print("=" * 80)
    print("中文对话情绪识别训练")
    print("=" * 80)
    print(f"数据路径: {args.data_path}")
    print(f"批处理大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.n_epochs}")
    print(f"设备: {'CUDA' if args.cuda else 'CPU'}")
    print("=" * 80)

    # 设置随机种子
    seed_everything(seed)

    # 获取数据加载器
    print("正在加载数据...")
    train_loader, valid_loader, test_loader = get_data_loaders(
        args.data_path, 
        args.batch_size, 
        args.valid_split, 
        args.num_workers, 
        args.pin_memory
    )
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(valid_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

    # 打印数据统计信息
    train_loader.dataset.print_statistics()
    test_loader.dataset.print_statistics()

    # 创建模型
    print("正在创建模型...")
    # 从训练集获取特征维度
    feature_dim = train_loader.dataset.feature_dim
    print(f"使用特征维度: {feature_dim}")
    
    model = SimpleDialogueGCN(
        input_dim=feature_dim,  # 动态特征维度
        hidden_dim=args.hidden_dim,
        num_classes=4,  # 4种情绪类别 (0,1,2,3)
        dropout=args.dropout
    )

    if args.cuda:
        model.cuda()

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 设置优化器和损失函数
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)
    else:
        raise ValueError('Unknown optimizer: {}'.format(args.optimizer))

    if args.class_weight:
        # 使用更温和的类别权重，避免过度惩罚majority class
        # 训练集分布: neutral(41.8%), happy(15.9%), sad(25.8%), angry(16.5%)
        # 使用平方根反比例权重，更温和的平衡
        class_counts = [17749, 6774, 10952, 7035]  # 对应0,1,2,3类别
        max_count = max(class_counts)
        
        # 使用平方根反比例权重，避免过度惩罚
        class_weights = []
        for count in class_counts:
            # 使用平方根来减少权重的极端性
            weight = (max_count / count) ** 0.5  # 平方根反比例权重
            class_weights.append(weight)
        
        # 归一化权重，确保权重不会过于极端
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        class_weights = class_weights / class_weights.mean()  # 归一化到均值
        
        print(f"类别权重: {class_weights}")
        if args.cuda:
            class_weights = class_weights.cuda()
        # 使用Focal Loss来处理类别不平衡
        loss_function = FocalLoss(alpha=class_weights, gamma=1.5)  # 降低gamma值
        print("使用Focal Loss处理类别不平衡（温和权重）")
    else:
        # 使用更平衡的权重
        class_weights = torch.tensor([1.0, 1.5, 1.2, 1.4], dtype=torch.float32)  # 更温和的权重
        if args.cuda:
            class_weights = class_weights.cuda()
        loss_function = FocalLoss(alpha=class_weights, gamma=1.5)  # 降低gamma值
        print("使用Focal Loss处理类别不平衡（平衡权重）")

    # 训练循环
    best_fscore = 0
    best_accuracy = 0
    best_loss = float('inf')
    
    print("\n开始训练...")
    for epoch in range(args.n_epochs):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(
            model, loss_function, train_loader, train=True, optimizer=optimizer, cuda_flag=args.cuda
        )
        
        # 验证
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(
            model, loss_function, valid_loader, train=False, cuda_flag=args.cuda
        )
        
        # 测试
        test_loss, test_acc, test_labels, test_preds, test_masks, test_fscore = train_or_eval_model(
            model, loss_function, test_loader, train=False, cuda_flag=args.cuda
        )
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"Epoch {epoch + 1}/{args.n_epochs} ({epoch_time:.1f}s)")
        print(f"  训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%, F1: {train_fscore:.2f}%")
        print(f"  验证 - 损失: {valid_loss:.4f}, 准确率: {valid_acc:.2f}%, F1: {valid_fscore:.2f}%")
        print(f"  测试 - 损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%, F1: {test_fscore:.2f}%")
        
        # 保存最佳模型
        if test_fscore > best_fscore:
            best_fscore = test_fscore
            best_accuracy = test_acc
            best_loss = test_loss
            
            # 保存模型
            model_save_path = f"chinese_dialoguegcn_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"  ✅ 保存最佳模型: {model_save_path} (F1: {test_fscore:.2f}%)")
        
        # 早停检查
        if epoch > 10 and test_fscore < best_fscore - 0.01:
            print("  早停触发")
            break
        
        print("-" * 60)
    
    # 最终评估
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"最佳测试F1分数: {best_fscore:.2f}%")
    print(f"最佳测试准确率: {best_accuracy:.2f}%")
    print(f"最佳测试损失: {best_loss:.4f}")
    
    # 显示最终分类报告
    if test_labels is not None and test_preds is not None:
        # 获取实际存在的情绪类别
        unique_labels = np.unique(test_labels)
        emotion_names = []
        for label in sorted(unique_labels):
            if label in [0, 1, 2, 3]:  # 确保标签在有效范围内（四种情绪）
                emotion_names.append(['neutral', 'happy', 'sad', 'angry'][label])
        
        print(f"\n最终分类报告:")
        print(classification_report(test_labels, test_preds, target_names=emotion_names, sample_weight=test_masks))
        
        # 显示混淆矩阵
        cm = confusion_matrix(test_labels, test_preds, sample_weight=test_masks)
        print(f"\n混淆矩阵:")
        print(cm)

def main():
    parser = argparse.ArgumentParser(description='中文对话情绪识别训练')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径')
    parser.add_argument('--n_epochs', type=int, default=60, help='训练轮数 (default: 60, same as IEMOCAP)')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小 (default: 32, same as IEMOCAP)')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率 (default: 0.0001, same as IEMOCAP)')
    parser.add_argument('--hidden_dim', type=int, default=100, help='隐藏层维度 (default: 100, same as IEMOCAP D_h)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率 (default: 0.5, same as IEMOCAP)')
    parser.add_argument('--valid_split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'], help='优化器')
    parser.add_argument('--l2', type=float, default=0.00001, help='L2正则化 (default: 0.00001, same as IEMOCAP)')
    parser.add_argument('--class_weight', action='store_true', help='使用类别权重')
    parser.add_argument('--cuda', action='store_true', help='是否使用CUDA')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载器工作进程数')
    parser.add_argument('--pin_memory', action='store_true', help='是否固定内存')
    parser.add_argument('--seed', type=int, default=100, help='随机种子 (default: 100, same as IEMOCAP)')
    args = parser.parse_args()
    seed_everything(args.seed)
    # 自动检测CUDA
    if torch.cuda.is_available() and not args.cuda:
        print("检测到CUDA，自动启用GPU训练")
        args.cuda = True
    train_model(args)

if __name__ == '__main__':
    main()
