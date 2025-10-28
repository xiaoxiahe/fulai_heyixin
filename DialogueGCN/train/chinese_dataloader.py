import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

class ChineseDataset(Dataset):
    """
    中文多轮对话情绪识别数据集加载器
    适配DialogueGCN模型
    """
    
    def __init__(self, pkl_path, split='train'):
        """
        初始化数据集
        
        Args:
            pkl_path: pkl文件路径
            split: 数据集分割 ('train', 'valid', 'test')
        """
        # 加载pkl文件
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 检查数据格式并提取数据
        if isinstance(data, tuple) and len(data) >= 6:
            if len(data) == 6:
                # 旧格式: (video_ids, video_speakers, video_labels, video_text, trainVids, testVids)
                print("检测到元组格式数据（6元组），正在转换...")
                video_ids, video_speakers, video_labels, video_text, trainVids, testVids = data
                devVids = []  # 没有验证集
            elif len(data) == 7:
                # 新格式: (video_ids, video_speakers, video_labels, video_text, trainVids, devVids, testVids)
                print("检测到元组格式数据（7元组），正在转换...")
                video_ids, video_speakers, video_labels, video_text, trainVids, devVids, testVids = data
            
            # 转换为字典格式以保持兼容性
            self.video_ids = video_ids
            self.video_speakers = video_speakers
            self.video_labels = video_labels
            self.video_text = video_text
            self.video_audio = {}  # 创建空的音频特征字典
            self.video_visual = {}  # 创建空的视觉特征字典
            self.video_sentence = {}  # 创建空的句子字典
            self.train_ids = trainVids
            self.valid_ids = devVids  # 使用验证集
            self.test_ids = testVids
            
            # 为每个视频ID创建空的音频和视觉特征
            for vid in video_ids:
                if vid not in self.video_audio:
                    self.video_audio[vid] = []
                if vid not in self.video_visual:
                    self.video_visual[vid] = []
                if vid not in self.video_sentence:
                    self.video_sentence[vid] = []
                    
        elif isinstance(data, dict):
            # 字典格式
            print("检测到字典格式数据")
            self.video_ids = data['video_ids']
            self.video_speakers = data['video_speakers'] 
            self.video_labels = data['video_labels']
            self.video_text = data['video_text']
            self.video_audio = data.get('video_audio', {})
            self.video_visual = data.get('video_visual', {})
            self.video_sentence = data.get('video_sentence', {})
            self.train_ids = data['train_ids']
            self.valid_ids = data.get('valid_ids', [])
            self.test_ids = data['test_ids']
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}")
        
        # 根据split选择数据
        if split == 'train':
            if len(self.valid_ids) > 0:
                # 有验证集，使用全部训练集
                self.keys = self.train_ids
            else:
                # 没有验证集，从训练集中分割80%作为训练集
                print("没有验证集，从训练集中分割80%作为训练集")
                train_size = int(0.8 * len(self.train_ids))
                self.keys = self.train_ids[:train_size]
        elif split == 'valid':
            if len(self.valid_ids) > 0:
                self.keys = self.valid_ids
            else:
                # 如果没有验证集，从训练集中分割20%作为验证集
                print("没有验证集，从训练集中分割20%作为验证集")
                train_size = int(0.8 * len(self.train_ids))
                self.keys = self.train_ids[train_size:]
        elif split == 'test':
            self.keys = self.test_ids
        else:
            raise ValueError("split must be 'train', 'valid', or 'test'")
        
        self.len = len(self.keys)
        
        # 情绪标签映射 - 与train_chinese_auto_dim.py中的n_classes=6保持一致
        # 6种情绪类别
        self.emotion_mapping = {
            'neutral': 0,
            'happy': 1, 
            'sad': 2,
            'angry': 3,
            'fear': 4,
            'surprise': 5,
            # 添加可能的其他标签映射（以防数据中有不同的表示）
            0: 0,  # 如果标签已经是数字
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5
        }
        
        # 说话者映射 - 支持4个说话者
        self.speaker_mapping = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
            # 添加数字类型的映射（以防数据中是数字）
            0: 0,
            1: 1,
            2: 2,
            3: 3
        }
        self.n_speakers = 4  # 说话者数量
        self.n_emotions = 6   # 情绪类别数量
    
    def __getitem__(self, index):
        """获取单个样本"""
        vid = self.keys[index]
        
        # 获取文本特征 (假设已经是768维BERT特征)
        text_features = torch.FloatTensor(self.video_text[vid])
        
        # 获取音频特征 (如果没有，使用零向量)
        if vid in self.video_audio:
            audio_features = torch.FloatTensor(self.video_audio[vid])
        else:
            audio_features = torch.zeros(text_features.size(0), 100)
        
        # 获取视觉特征 (如果没有，使用零向量)
        if vid in self.video_visual:
            visual_features = torch.FloatTensor(self.video_visual[vid])
        else:
            visual_features = torch.zeros(text_features.size(0), 100)
        
        # 获取说话者信息
        speakers = self.video_speakers[vid]
        speaker_features = []
        for speaker in speakers:
            if speaker in self.speaker_mapping:
                speaker_id = self.speaker_mapping[speaker]
            elif isinstance(speaker, (int, float)):
                # 如果是数字，确保在有效范围内
                speaker_id = int(speaker)
                if speaker_id < 0 or speaker_id >= self.n_speakers:
                    speaker_id = 0  # 超出范围，默认为第一个说话者
            else:
                speaker_id = 0  # 未知说话者，默认为第一个说话者
            
            # 创建one-hot编码
            speaker_onehot = [0] * self.n_speakers
            speaker_onehot[speaker_id] = 1
            speaker_features.append(speaker_onehot)
        
        speaker_features = torch.FloatTensor(speaker_features)
        
        # 获取情绪标签
        labels = self.video_labels[vid]
        # 将情绪标签转换为数字，增强健壮性
        numeric_labels = []
        for label in labels:
            if label in self.emotion_mapping:
                label_id = self.emotion_mapping[label]
            elif isinstance(label, (int, float)):
                # 如果是数字，确保在有效范围内
                label_id = int(label)
                if label_id < 0 or label_id >= self.n_emotions:
                    label_id = 0  # 超出范围，默认为neutral
            else:
                label_id = 0  # 未知标签，默认为neutral
            numeric_labels.append(label_id)
        
        labels = torch.LongTensor(numeric_labels)
        
        # 创建mask (所有位置都是有效的)
        mask = torch.ones(len(labels))
        
        return text_features, visual_features, audio_features, speaker_features, mask, labels, vid
    
    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        """批处理函数"""
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in dat]


class ChineseDatasetWithBERT(Dataset):
    """
    带BERT特征提取的中文数据集加载器
    如果pkl文件中没有预提取的BERT特征，使用此类
    """
    
    def __init__(self, pkl_path, split='train', bert_model_name='hfl/chinese-roberta-wwm-ext'):
        """
        初始化数据集
        
        Args:
            pkl_path: pkl文件路径
            split: 数据集分割 ('train', 'valid', 'test')
            bert_model_name: BERT模型名称
        """
        # 加载pkl文件
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 提取数据
        self.video_ids = data['video_ids']
        self.video_speakers = data['video_speakers'] 
        self.video_labels = data['video_labels']
        self.video_sentence = data['video_sentence']  # 原始文本
        self.train_ids = data['train_ids']
        self.valid_ids = data['valid_ids'] 
        self.test_ids = data['test_ids']
        
        # 根据split选择数据
        if split == 'train':
            self.keys = self.train_ids
        elif split == 'valid':
            self.keys = self.valid_ids
        elif split == 'test':
            self.keys = self.test_ids
        else:
            raise ValueError("split must be 'train', 'valid', or 'test'")
        
        self.len = len(self.keys)
        
        # 初始化BERT模型
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.bert_model.eval()
        
        # 情绪标签映射
        self.emotion_mapping = {
            'neutral': 0,
            'happy': 1, 
            'sad': 2,
            'angry': 3,
            'fear': 4,
            'surprise': 5
        }
        
        # 说话者映射
        self.speaker_mapping = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3
        }
    
    def _extract_bert_features(self, text):
        """提取BERT特征"""
        with torch.no_grad():
            # 分词
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            
            # 获取BERT输出
            outputs = self.bert_model(**inputs)
            
            # 使用[CLS]标记的表示作为句子特征
            features = outputs.last_hidden_state[:, 0, :]  # [1, 768]
            
            return features.squeeze(0)  # [768]
    
    def __getitem__(self, index):
        """获取单个样本"""
        vid = self.keys[index]
        
        # 获取原始文本
        sentences = self.video_sentence[vid]
        
        # 提取BERT特征
        text_features = []
        for sentence in sentences:
            features = self._extract_bert_features(sentence)
            text_features.append(features)
        
        text_features = torch.stack(text_features)  # [seq_len, 768]
        
        # 创建音频和视觉特征 (零向量)
        audio_features = torch.zeros(text_features.size(0), 100)
        visual_features = torch.zeros(text_features.size(0), 100)
        
        # 获取说话者信息
        speakers = self.video_speakers[vid]
        speaker_features = []
        for speaker in speakers:
            if speaker in self.speaker_mapping:
                speaker_id = self.speaker_mapping[speaker]
            elif isinstance(speaker, (int, float)):
                # 如果是数字，确保在有效范围内
                speaker_id = int(speaker)
                if speaker_id < 0 or speaker_id >= self.n_speakers:
                    speaker_id = 0  # 超出范围，默认为第一个说话者
            else:
                speaker_id = 0  # 未知说话者，默认为第一个说话者
            
            # 创建one-hot编码
            speaker_onehot = [0] * self.n_speakers
            speaker_onehot[speaker_id] = 1
            speaker_features.append(speaker_onehot)
        
        speaker_features = torch.FloatTensor(speaker_features)
        
        # 获取情绪标签
        labels = self.video_labels[vid]
        # 将情绪标签转换为数字，增强健壮性
        numeric_labels = []
        for label in labels:
            if label in self.emotion_mapping:
                label_id = self.emotion_mapping[label]
            elif isinstance(label, (int, float)):
                # 如果是数字，确保在有效范围内
                label_id = int(label)
                if label_id < 0 or label_id >= self.n_emotions:
                    label_id = 0  # 超出范围，默认为neutral
            else:
                label_id = 0  # 未知标签，默认为neutral
            numeric_labels.append(label_id)
        
        labels = torch.LongTensor(numeric_labels)
        
        # 创建mask (所有位置都是有效的)
        mask = torch.ones(len(labels))
        
        return text_features, visual_features, audio_features, speaker_features, mask, labels, vid
    
    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        """批处理函数"""
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in dat]
