#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两人对话中文情绪识别训练启动脚本
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='两人对话中文情绪识别训练启动脚本')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--gpu', action='store_true', help='使用GPU训练')
    
    args = parser.parse_args()
    
    # 检查数据文件
    if not os.path.exists(args.data_path):
        print(f"❌ 数据文件不存在: {args.data_path}")
        return
    
    print("=" * 80)
    print("两人对话中文情绪识别训练")
    print("=" * 80)
    print(f"数据文件: {args.data_path}")
    print(f"训练轮数: {args.epochs}")
    print(f"批处理大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"使用GPU: {'是' if args.gpu else '否'}")
    print("=" * 80)
    
    # 构建训练命令
    cmd = [
        'python', 'train_two_person_chinese.py',
        '--data_path', args.data_path,
        '--n_epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--base_model', 'DialogRNN',
        '--n_classes', '7',
        '--valid_split', '0.1',
        '--optimizer', 'adam',
        '--l2', '0.00001',
        '--dropout', '0.5',
        '--dropout_rec', '0.5',
        '--nodal_attention',
        '--class_weight'
    ]
    
    if args.gpu:
        cmd.append('--cuda')
    
    print(f"执行命令: {' '.join(cmd)}")
    print("=" * 80)
    
    # 运行训练
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n🎉 训练完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        return
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        return

if __name__ == "__main__":
    main()
