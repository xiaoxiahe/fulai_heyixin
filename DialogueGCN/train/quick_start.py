#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文多轮对话情绪识别快速启动脚本
一键完成数据预处理和模型训练
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*50}")
    print(f"正在执行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*50}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✅ {description} 完成 (耗时: {end_time - start_time:.2f}秒)")
        if result.stdout:
            print("输出:")
            print(result.stdout)
    else:
        print(f"❌ {description} 失败")
        print("错误信息:")
        print(result.stderr)
        return False
    
    return True

def check_file_exists(file_path, description):
    """检查文件是否存在"""
    if not os.path.exists(file_path):
        print(f"❌ 错误: {description} 不存在: {file_path}")
        return False
    print(f"✅ {description} 存在: {file_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='中文多轮对话情绪识别快速启动')
    
    # 必需参数
    parser.add_argument('--input', required=True, help='输入CSV文件路径')
    
    # 可选参数
    parser.add_argument('--output', default='chinese_data.pkl', help='输出pkl文件路径')
    parser.add_argument('--model', default='LSTM', choices=['LSTM', 'GRU', 'DialogRNN'], help='基础模型')
    parser.add_argument('--use_graph', action='store_true', help='使用图神经网络')
    parser.add_argument('--use_bert', action='store_true', help='使用BERT特征提取')
    parser.add_argument('--epochs', type=int, default=60, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--output_dir', default='./saved/chinese/', help='模型保存目录')
    parser.add_argument('--skip_preprocess', action='store_true', help='跳过数据预处理步骤')
    parser.add_argument('--skip_analysis', action='store_true', help='跳过数据分析步骤')
    
    args = parser.parse_args()
    
    print("🚀 中文多轮对话情绪识别快速启动")
    print("="*60)
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"基础模型: {args.model}")
    print(f"使用图神经网络: {args.use_graph}")
    print(f"使用BERT: {args.use_bert}")
    print(f"训练轮数: {args.epochs}")
    print(f"批处理大小: {args.batch_size}")
    print("="*60)
    
    # 检查输入文件
    if not check_file_exists(args.input, "输入CSV文件"):
        return
    
    # 检查必要文件
    required_files = [
        'preprocess_chinese_data.py',
        'emotion_mapping.py', 
        'train_chinese.py',
        'chinese_dataloader.py',
        'model.py'
    ]
    
    for file in required_files:
        if not check_file_exists(file, f"必要文件 {file}"):
            print(f"请确保所有必要文件都在当前目录下")
            return
    
    # 步骤1: 数据分析 (可选)
    if not args.skip_analysis:
        cmd = f"python emotion_mapping.py --input {args.input} --emotion_col Emotion --analyze --weights"
        if not run_command(cmd, "数据分析"):
            print("⚠️ 数据分析失败，但可以继续")
    
    # 步骤2: 数据预处理 (可选)
    if not args.skip_preprocess:
        cmd = f"python preprocess_chinese_data.py --input {args.input} --output {args.output}"
        if args.use_bert:
            cmd += " --bert_model hfl/chinese-roberta-wwm-ext"
        cmd += f" --batch_size {args.batch_size}"
        
        if not run_command(cmd, "数据预处理"):
            print("❌ 数据预处理失败，无法继续")
            return
    else:
        if not check_file_exists(args.output, "pkl数据文件"):
            print("❌ 跳过数据预处理但pkl文件不存在，无法继续")
            return
    
    # 步骤3: 模型训练
    cmd = f"python train_chinese.py --data_path {args.output} --base_model {args.model}"
    cmd += f" --epochs {args.epochs} --batch_size {args.batch_size}"
    cmd += f" --output_dir {args.output_dir}"
    
    if args.use_graph:
        cmd += " --graph_model --nodal_attention"
    
    if args.use_bert:
        cmd += " --use_bert"
        # BERT需要更小的batch_size
        if args.batch_size > 16:
            cmd = cmd.replace(f"--batch_size {args.batch_size}", "--batch_size 16")
            print("⚠️ 使用BERT时自动调整batch_size为16")
    
    if not run_command(cmd, "模型训练"):
        print("❌ 模型训练失败")
        return
    
    # 完成
    print("\n🎉 训练完成!")
    print("="*60)
    print(f"模型已保存到: {args.output_dir}")
    print(f"最佳模型文件: best_model_*.pkl")
    print(f"训练历史: training_history.json")
    print("="*60)
    
    # 显示结果文件
    if os.path.exists(args.output_dir):
        print("\n📁 生成的文件:")
        for file in os.listdir(args.output_dir):
            file_path = os.path.join(args.output_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  {file} ({size/1024/1024:.2f} MB)")

if __name__ == '__main__':
    main()
