#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 full_data_cleaned.csv 中的情绪标签映射为英文
根据 gpu_processor.py 的映射关系
"""

import pandas as pd
import os

def create_emotion_mapping():
    """
    创建情绪映射字典
    从中文/原始标签映射到标准英文标签
    
    根据 gpu_processor.py 的映射:
    - relaxed, happy, grateful, positive-other -> happy (1)
    - neutral -> neutral (0)
    - astonished (surprise) -> neutral (0)
    - depress, fear, negative-other, sadness, worried -> sad (2)
    - anger, disgust -> angry (3)
    """
    
    emotion_to_english = {
        # 映射为 happy
        'relaxed': 'happy',
        'happy': 'happy',
        'grateful': 'happy',
        'positive-other': 'happy',
        
        # 映射为 neutral
        'neutral': 'neutral',
        'astonished': 'neutral',  # surprise -> neutral
        'surprise': 'neutral',
        
        # 映射为 sad
        'depress': 'sad',
        'fear': 'sad',
        'negative-other': 'sad',
        'sadness': 'sad',
        'worried': 'sad',
        'sad': 'sad',
        
        # 映射为 angry
        'anger': 'angry',
        'disgust': 'angry',
        'angry': 'angry',
    }
    
    return emotion_to_english

def map_emotions_in_csv(input_file, output_file=None):
    """
    处理CSV文件，将情绪标签映射为英文
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径（如果为None，则覆盖原文件）
    """
    
    print(f"📂 读取文件: {input_file}")
    
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    print(f"✅ 文件加载成功，共 {len(df)} 行")
    print(f"📋 列名: {list(df.columns)}")
    
    # 检查是否有Emotion列
    if 'Emotion' not in df.columns:
        print("❌ 错误：文件中没有 'Emotion' 列")
        return
    
    # 显示原始情绪分布
    print(f"\n📊 原始情绪分布:")
    emotion_counts = df['Emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count}")
    
    # 创建映射字典
    emotion_mapping = create_emotion_mapping()
    
    # 保存原始列以便对比
    df['Emotion_Original'] = df['Emotion'].copy()
    
    # 映射情绪标签
    print(f"\n🔄 正在映射情绪标签...")
    
    unmapped_emotions = set()
    mapped_count = 0
    
    def map_emotion(emotion):
        nonlocal mapped_count, unmapped_emotions
        
        if pd.isna(emotion):
            return emotion
        
        emotion_lower = str(emotion).lower().strip()
        
        if emotion_lower in emotion_mapping:
            mapped_count += 1
            return emotion_mapping[emotion_lower]
        else:
            unmapped_emotions.add(str(emotion))
            return emotion  # 保持原样
    
    df['Emotion'] = df['Emotion_Original'].apply(map_emotion)
    
    # 显示映射结果
    print(f"✅ 映射完成！")
    print(f"   成功映射: {mapped_count} 个标签")
    
    if unmapped_emotions:
        print(f"\n⚠️ 未映射的情绪标签 ({len(unmapped_emotions)} 个):")
        for emotion in sorted(unmapped_emotions):
            print(f"   - {emotion}")
    
    # 显示映射后的情绪分布
    print(f"\n📊 映射后情绪分布:")
    new_emotion_counts = df['Emotion'].value_counts()
    for emotion, count in new_emotion_counts.items():
        print(f"   {emotion}: {count}")
    
    # 显示映射示例
    print(f"\n🔍 映射示例（前10行）:")
    print(df[['Emotion_Original', 'Emotion']].head(10).to_string())
    
    # 保存文件
    if output_file is None:
        output_file = input_file.replace('.csv', '_mapped.csv')
    
    # 删除临时的原始列（可选：保留用于检查）
    # df = df.drop('Emotion_Original', axis=1)
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 文件已保存: {output_file}")
    print(f"📏 输出文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    return df

def main():
    """主函数"""
    print("=" * 60)
    print("情绪标签映射工具")
    print("=" * 60)
    
    input_file = 'full_data_cleaned.csv'
    output_file = 'full_data_cleaned_map.csv'
    
    if not os.path.exists(input_file):
        print(f"❌ 错误：找不到输入文件 {input_file}")
        return
    
    # 处理文件
    df = map_emotions_in_csv(input_file, output_file)
    
    print("\n" + "=" * 60)
    print("✅ 处理完成！")
    print("=" * 60)
    
    # 显示映射规则
    print("\n📋 使用的映射规则:")
    print("   happy <- relaxed, happy, grateful, positive-other")
    print("   neutral <- neutral, astonished, surprise")
    print("   sad <- depress, fear, negative-other, sadness, worried")
    print("   angry <- anger, disgust")

if __name__ == "__main__":
    main()

