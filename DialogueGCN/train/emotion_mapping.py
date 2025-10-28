#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文情绪标签映射工具
用于将原始情绪标签映射到DialogueGCN可识别的标准情绪标签
"""

import pandas as pd
import numpy as np
from collections import Counter
import argparse

class ChineseEmotionMapper:
    """中文情绪标签映射器"""
    
    def __init__(self):
        """初始化情绪映射器"""
        # 标准情绪标签映射 (5类，将surprise归类到neutral)
        self.standard_emotions = {
            'neutral': 0,
            'happy': 1, 
            'sad': 2,
            'angry': 3,
            'fear': 4
        }
        
        # 反向映射
        self.id_to_emotion = {v: k for k, v in self.standard_emotions.items()}
        
        # 原始情绪标签到标准标签的映射规则
        self.emotion_mapping_rules = {
            # 中性情绪
            'neutral': 'neutral',
            'neutrality': 'neutral',
            '无情绪': 'neutral',
            '平静': 'neutral',
            
            # 快乐情绪
            'happy': 'happy',
            'happiness': 'happy',
            'joy': 'happy',
            '高兴': 'happy',
            '开心': 'happy',
            '快乐': 'happy',
            '兴奋': 'happy',
            'relaxed': 'happy',
            'grateful': 'happy',
            'positive-other': 'happy',
            
            # 悲伤情绪
            'sad': 'sad',
            'sadness': 'sad',
            'sorrow': 'sad',
            '悲伤': 'sad',
            '难过': 'sad',
            'depress': 'sad',
            'depression': 'sad',
            'depressed': 'sad',
            'negative-other': 'sad',
            'worried': 'sad',
            'worry': 'sad',
            '担心': 'sad',
            
            # 愤怒情绪
            'angry': 'angry',
            'anger': 'angry',
            'mad': 'angry',
            '愤怒': 'angry',
            '生气': 'angry',
            '恼火': 'angry',
            'disgust': 'angry',
            'disgusted': 'angry',
            '厌恶': 'angry',
            
            # 恐惧情绪
            'fear': 'fear',
            'afraid': 'fear',
            'scared': 'fear',
            '恐惧': 'fear',
            '害怕': 'fear',
            '担心': 'fear',
            
            # 惊讶情绪 -> 归类到neutral
            'surprise': 'neutral',
            'surprised': 'neutral',
            'astonished': 'neutral',
            '惊讶': 'neutral',
            '吃惊': 'neutral',
            '意外': 'neutral'
        }
    
    def map_emotion(self, emotion):
        """
        将原始情绪标签映射到标准标签
        
        Args:
            emotion: 原始情绪标签
            
        Returns:
            str: 标准情绪标签
        """
        if pd.isna(emotion) or emotion == '':
            return 'neutral'
        
        # 转换为小写并去除空格
        emotion = str(emotion).lower().strip()
        
        # 直接映射
        if emotion in self.emotion_mapping_rules:
            return self.emotion_mapping_rules[emotion]
        
        # 模糊匹配
        for original, standard in self.emotion_mapping_rules.items():
            if original in emotion or emotion in original:
                return standard
        
        # 默认返回中性
        print(f"警告: 未知情绪标签 '{emotion}'，映射为 'neutral'")
        return 'neutral'
    
    def analyze_emotions(self, csv_path, emotion_col='Emotion'):
        """
        分析数据集中的情绪分布
        
        Args:
            csv_path: CSV文件路径
            emotion_col: 情绪列名
        """
        print(f"正在分析情绪分布: {csv_path}")
        
        # 读取数据
        df = pd.read_csv(csv_path)
        
        if emotion_col not in df.columns:
            print(f"错误: 找不到情绪列 '{emotion_col}'")
            print(f"可用列: {df.columns.tolist()}")
            return
        
        # 统计原始情绪分布
        original_emotions = df[emotion_col].value_counts()
        print(f"\n原始情绪分布 (共{len(original_emotions)}种):")
        for emotion, count in original_emotions.items():
            print(f"  {emotion}: {count}")
        
        # 映射后的情绪分布
        mapped_emotions = df[emotion_col].apply(self.map_emotion)
        mapped_counts = mapped_emotions.value_counts()
        
        print(f"\n映射后情绪分布:")
        for emotion in self.standard_emotions.keys():
            count = mapped_counts.get(emotion, 0)
            percentage = count / len(df) * 100
            print(f"  {emotion}: {count} ({percentage:.2f}%)")
        
        # 显示映射详情
        print(f"\n映射详情:")
        mapping_details = {}
        for original in original_emotions.index:
            mapped = self.map_emotion(original)
            if mapped not in mapping_details:
                mapping_details[mapped] = []
            mapping_details[mapped].append(original)
        
        for standard, originals in mapping_details.items():
            print(f"  {standard}: {', '.join(originals)}")
        
        return mapped_emotions
    
    def create_mapping_file(self, csv_path, emotion_col='Emotion', output_path='emotion_mapping.json'):
        """
        创建情绪映射文件
        
        Args:
            csv_path: CSV文件路径
            emotion_col: 情绪列名
            output_path: 输出映射文件路径
        """
        print(f"正在创建情绪映射文件: {output_path}")
        
        # 读取数据
        df = pd.read_csv(csv_path)
        
        if emotion_col not in df.columns:
            print(f"错误: 找不到情绪列 '{emotion_col}'")
            return
        
        # 获取所有唯一的原始情绪标签
        unique_emotions = df[emotion_col].unique()
        
        # 创建映射字典
        mapping_dict = {}
        for emotion in unique_emotions:
            if pd.notna(emotion):
                mapped = self.map_emotion(emotion)
                mapping_dict[str(emotion)] = mapped
        
        # 保存映射文件
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_dict, f, ensure_ascii=False, indent=2)
        
        print(f"情绪映射文件已保存: {output_path}")
        print(f"映射了 {len(mapping_dict)} 种情绪标签")
    
    def get_emotion_weights(self, csv_path, emotion_col='Emotion'):
        """
        计算情绪类别权重 (用于处理类别不平衡)
        
        Args:
            csv_path: CSV文件路径
            emotion_col: 情绪列名
            
        Returns:
            dict: 情绪权重字典
        """
        # 读取数据
        df = pd.read_csv(csv_path)
        
        if emotion_col not in df.columns:
            print(f"错误: 找不到情绪列 '{emotion_col}'")
            return None
        
        # 映射情绪标签
        mapped_emotions = df[emotion_col].apply(self.map_emotion)
        
        # 计算每个类别的数量
        emotion_counts = mapped_emotions.value_counts()
        
        # 计算权重 (使用逆频率)
        total_count = len(mapped_emotions)
        weights = {}
        
        for emotion in self.standard_emotions.keys():
            count = emotion_counts.get(emotion, 1)  # 避免除零
            weight = total_count / (len(self.standard_emotions) * count)
            weights[emotion] = weight
        
        print("情绪类别权重:")
        for emotion, weight in weights.items():
            print(f"  {emotion}: {weight:.4f}")
        
        return weights


def main():
    parser = argparse.ArgumentParser(description='中文情绪标签映射工具')
    parser.add_argument('--input', required=True, help='输入CSV文件路径')
    parser.add_argument('--emotion_col', default='Emotion', help='情绪列名')
    parser.add_argument('--output', default='emotion_mapping.json', help='输出映射文件路径')
    parser.add_argument('--analyze', action='store_true', help='分析情绪分布')
    parser.add_argument('--weights', action='store_true', help='计算类别权重')
    
    args = parser.parse_args()
    
    # 创建映射器
    mapper = ChineseEmotionMapper()
    
    # 分析情绪分布
    if args.analyze:
        mapper.analyze_emotions(args.input, args.emotion_col)
    
    # 创建映射文件
    mapper.create_mapping_file(args.input, args.emotion_col, args.output)
    
    # 计算类别权重
    if args.weights:
        mapper.get_emotion_weights(args.input, args.emotion_col)


if __name__ == '__main__':
    main()
