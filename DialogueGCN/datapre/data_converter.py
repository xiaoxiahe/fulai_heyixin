# 数据格式转换工具
# 用于将不同格式的对话数据转换为DialogueGCN可训练的格式

import pandas as pd
import numpy as np
import pickle
import torch
import jieba
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import re
from collections import defaultdict
import random
import os
import json

class DataConverter:
    def __init__(self, model_name='hfl/chinese-roberta-wwm-ext'):
        """
        初始化数据转换器
        
        Args:
            model_name: RoBERTa模型名称
        """
        self.model_name = model_name
        print(f"正在加载RoBERTa模型: {model_name}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name)
        # 强制使用CPU，避免GPU内存不足
        self.device = torch.device('cpu')
        self.roberta.to(self.device)
        self.roberta.eval()
        print(f"RoBERTa模型加载完成，使用设备: {self.device}")
        
        # 情绪标签映射（CPED数据集13种情绪）
        self.emotion_map = {
            # 映射为happy (1)
            'relaxed': 1,
            'happy': 1,
            'grateful': 1,
            'positive-other': 1,
            
            # 映射为neutral (0)
            'neutral': 0,
            
            # 映射为surprise (5)
            'astonished': 5,
            
            # 映射为sad (2)
            'depress': 2,
            'fear': 2,
            'negative-other': 2,
            'sadness': 2,
            'worried': 2,
            
            # 映射为angry (3)
            'anger': 3,
            'disgust': 3,
        }
        
        # 说话者映射
        self.speaker_map = {}
        self.speaker_counter = 0
    
    def clean_text(self, text):
        """清洗文本"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        # 去除特殊字符，保留中文、英文、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        # 去除多余空格
        text = ' '.join(text.split())
        return text
    
    def get_speaker_id(self, speaker_name):
        """获取说话者ID"""
        if speaker_name not in self.speaker_map:
            speaker_id = chr(ord('A') + self.speaker_counter)
            self.speaker_map[speaker_name] = speaker_id
            self.speaker_counter += 1
        return self.speaker_map[speaker_name]
    
    def extract_roberta_features(self, text):
        """提取RoBERTa特征"""
        if not text or text.strip() == '':
            return np.zeros(768)
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.roberta(**inputs)
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return features[0]
        except Exception as e:
            print(f"特征提取失败: {text[:50]}... 错误: {e}")
            return np.zeros(768)
    
    def convert_json_to_iemocap(self, json_file, output_file):
        """
        将JSON格式的对话数据转换为IEMOCAP格式
        
        Args:
            json_file: 输入JSON文件路径
            output_file: 输出PKL文件路径
        """
        print(f"正在转换JSON数据: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 初始化数据结构
        video_ids = []
        video_speakers = {}
        video_labels = {}
        video_text = {}
        video_audio = {}
        video_visual = {}
        video_sentence = {}
        
        valid_conversations = 0
        
        for conv_id, conversation in tqdm(data.items(), desc="转换对话"):
            if 'utterances' not in conversation:
                continue
            
            utterances = conversation['utterances']
            if len(utterances) < 2:
                continue
            
            speakers = []
            labels = []
            text_features = []
            sentences = []
            
            for utterance in utterances:
                # 清洗文本
                text = self.clean_text(utterance.get('text', ''))
                if not text:
                    continue
                
                # 获取说话者ID
                speaker_name = utterance.get('speaker', 'Unknown')
                speaker_id = self.get_speaker_id(speaker_name)
                
                # 获取情绪标签
                emotion = utterance.get('emotion', 'neutral').lower()
                emotion_id = self.emotion_map.get(emotion, 0)
                
                # 提取BERT特征
                features = self.extract_roberta_features(text)
                
                speakers.append(speaker_id)
                labels.append(emotion_id)
                text_features.append(features)
                sentences.append(text)
            
            # 检查对话是否有效
            if len(speakers) < 2 or len(set(speakers)) < 2:
                continue
            
            video_ids.append(conv_id)
            video_speakers[conv_id] = speakers
            video_labels[conv_id] = labels
            video_text[conv_id] = text_features
            video_sentence[conv_id] = sentences
            video_audio[conv_id] = [np.zeros(100) for _ in range(len(speakers))]
            video_visual[conv_id] = [np.zeros(100) for _ in range(len(speakers))]
            
            valid_conversations += 1
        
        print(f"有效对话数: {valid_conversations}")
        
        # 划分数据集
        random.shuffle(video_ids)
        train_size = int(len(video_ids) * 0.7)
        dev_size = int(len(video_ids) * 0.15)
        
        trainVids = video_ids[:train_size]
        dev_vids = video_ids[train_size:train_size + dev_size]
        test_vids = video_ids[train_size + dev_size:]
        
        print(f"训练集: {len(trainVids)} 个对话")
        print(f"验证集: {len(dev_vids)} 个对话")
        print(f"测试集: {len(test_vids)} 个对话")
        
        # 保存数据
        data = (video_ids, video_speakers, video_labels, video_text,
                video_audio, video_visual, video_sentence, trainVids, test_vids)
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"数据保存完成: {output_file}")
        
        # 保存说话者映射
        speaker_map_file = output_file.replace('.pkl', '_speaker_map.pkl')
        with open(speaker_map_file, 'wb') as f:
            pickle.dump(self.speaker_map, f)
        
        return data
    
    def convert_csv_to_iemocap(self, csv_file, output_file, 
                              conv_id_col='TV_ID', 
                              text_col='Utterance',
                              speaker_col='Speaker',
                              emotion_col='Emotion'):
        """
        将CSV格式的对话数据转换为IEMOCAP格式
        
        Args:
            csv_file: 输入CSV文件路径
            output_file: 输出PKL文件路径
            conv_id_col: 对话ID列名
            text_col: 文本列名
            speaker_col: 说话者列名
            emotion_col: 情绪列名
        """
        print(f"正在转换CSV数据: {csv_file}")
        
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"数据加载完成，共 {len(df)} 条记录")
        
        # 按对话分组
        conversations = defaultdict(list)
        for idx, row in df.iterrows():
            conv_id = row[conv_id_col]
            conversations[conv_id].append(row)
        
        print(f"共找到 {len(conversations)} 个对话")
        
        # 初始化数据结构
        video_ids = []
        video_speakers = {}
        video_labels = {}
        video_text = {}
        video_audio = {}
        video_visual = {}
        video_sentence = {}
        
        valid_conversations = 0
        
        for conv_id, utterances in tqdm(conversations.items(), desc="转换对话"):
            # 按索引顺序排序
            utterances = sorted(utterances, key=lambda x: x.name)
            
            speakers = []
            labels = []
            text_features = []
            sentences = []
            
            for utterance in utterances:
                # 清洗文本
                text = self.clean_text(utterance[text_col])
                if not text:
                    continue
                
                # 获取说话者ID
                speaker_name = utterance[speaker_col]
                speaker_id = self.get_speaker_id(speaker_name)
                
                # 获取情绪标签
                emotion = utterance[emotion_col].lower() if pd.notna(utterance[emotion_col]) else 'neutral'
                emotion_id = self.emotion_map.get(emotion, 0)
                
                # 提取BERT特征
                features = self.extract_roberta_features(text)
                
                speakers.append(speaker_id)
                labels.append(emotion_id)
                text_features.append(features)
                sentences.append(text)
            
            # 检查对话是否有效
            if len(speakers) < 2 or len(set(speakers)) < 2:
                continue
            
            video_ids.append(conv_id)
            video_speakers[conv_id] = speakers
            video_labels[conv_id] = labels
            video_text[conv_id] = text_features
            video_sentence[conv_id] = sentences
            video_audio[conv_id] = [np.zeros(100) for _ in range(len(speakers))]
            video_visual[conv_id] = [np.zeros(100) for _ in range(len(speakers))]
            
            valid_conversations += 1
        
        print(f"有效对话数: {valid_conversations}")
        
        # 划分数据集
        random.shuffle(video_ids)
        train_size = int(len(video_ids) * 0.7)
        dev_size = int(len(video_ids) * 0.15)
        
        trainVids = video_ids[:train_size]
        dev_vids = video_ids[train_size:train_size + dev_size]
        test_vids = video_ids[train_size + dev_size:]
        
        print(f"训练集: {len(trainVids)} 个对话")
        print(f"验证集: {len(dev_vids)} 个对话")
        print(f"测试集: {len(test_vids)} 个对话")
        
        # 保存数据
        data = (video_ids, video_speakers, video_labels, video_text,
                video_audio, video_visual, video_sentence, trainVids, test_vids)
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"数据保存完成: {output_file}")
        
        # 保存说话者映射
        speaker_map_file = output_file.replace('.pkl', '_speaker_map.pkl')
        with open(speaker_map_file, 'wb') as f:
            pickle.dump(self.speaker_map, f)
        
        return data

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据格式转换工具')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--format', type=str, choices=['json', 'csv'], required=True, help='输入文件格式')
    parser.add_argument('--model', type=str, default='hfl/chinese-roberta-wwm-ext', help='RoBERTa模型名称')
    
    # CSV格式的列名参数
    parser.add_argument('--conv_id_col', type=str, default='TV_ID', help='对话ID列名')
    parser.add_argument('--text_col', type=str, default='Utterance', help='文本列名')
    parser.add_argument('--speaker_col', type=str, default='Speaker', help='说话者列名')
    parser.add_argument('--emotion_col', type=str, default='Emotion', help='情绪列名')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:  # 只有当输出目录不为空时才创建
        os.makedirs(output_dir, exist_ok=True)
    
    # 初始化转换器
    converter = DataConverter(model_name=args.model)
    
    # 根据格式进行转换
    if args.format == 'json':
        converter.convert_json_to_iemocap(args.input, args.output)
    elif args.format == 'csv':
        converter.convert_csv_to_iemocap(
            args.input, args.output,
            conv_id_col=args.conv_id_col,
            text_col=args.text_col,
            speaker_col=args.speaker_col,
            emotion_col=args.emotion_col
        )
    
    print("数据转换完成！")

if __name__ == "__main__":
    main()
