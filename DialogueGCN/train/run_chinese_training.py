#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文对话情绪识别训练启动脚本 - 增强版
支持gpu_processor_enhanced.py的特征提取配置
严格按照用户指定的4种情绪类别
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='中文对话情绪识别训练启动脚本 - 增强版')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./saved/optimized/', help='输出目录')
    
    # 基础训练参数
    parser.add_argument('--n_epochs', type=int, default=40, help='训练轮数 (default: 40)')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小 (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率 (default: 0.0001)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率 (default: 0.2)')
    parser.add_argument('--seed', type=int, default=100, help='随机种子 (default: 100)')
    
    # 模型架构参数
    parser.add_argument('--base_model', type=str, default='GRU', choices=['LSTM', 'GRU', 'DialogRNN'], help='基础模型类型')
    parser.add_argument('--graph_model', action='store_true', help='使用图神经网络')
    parser.add_argument('--nodal_attention', action='store_true', help='使用节点注意力')
    
    # 特征提取参数（与gpu_processor_enhanced.py一致）
    parser.add_argument('--feature_dim', type=int, default=2304, help='特征维度 (768=单一池化, 2304=多池化)')
    parser.add_argument('--pooling_strategy', type=str, default='multi', 
                       choices=['cls', 'mean', 'max', 'attention', 'multi'],
                       help='池化策略 (default: multi)')
    parser.add_argument('--use_context_window', action='store_true', help='使用上下文窗口')
    parser.add_argument('--context_window_size', type=int, default=8, help='上下文窗口大小')
    parser.add_argument('--max_length', type=int, default=256, help='BERT最大序列长度')
    
    # 对话窗口参数
    parser.add_argument('--windowp', type=int, default=5, help='过去窗口大小')
    parser.add_argument('--windowf', type=int, default=5, help='未来窗口大小')
    
    # 优化参数
    parser.add_argument('--l2', type=float, default=0.00001, help='L2正则化')
    parser.add_argument('--class_weight', action='store_true', help='使用类别权重')
    parser.add_argument('--cuda', action='store_true', help='是否使用CUDA')
    
    # 高级选项
    parser.add_argument('--use_recommended_config', action='store_true', help='使用推荐配置')

    args = parser.parse_args()

    # 应用推荐配置
    if args.use_recommended_config:
        print("使用推荐配置...")
        args.graph_model = True
        args.nodal_attention = True
        args.use_context_window = True
        args.feature_dim = 2304
        args.pooling_strategy = 'multi'
        args.max_length = 256
        args.windowp = 5
        args.windowf = 5
        args.dropout = 0.2
        args.lr = 0.0001

    # 检查数据文件
    if not os.path.exists(args.data_path):
        print(f"✗ 数据文件不存在: {args.data_path}")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("中文对话情绪识别训练 - 增强版")
    print("=" * 80)
    print(f"数据文件: {args.data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.n_epochs}")
    print(f"批处理大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"Dropout: {args.dropout}")
    print(f"随机种子: {args.seed}")
    print(f"使用GPU: {'是' if args.cuda else '否'}")
    print("=" * 80)
    
    print("模型架构配置:")
    print(f"  基础模型: {args.base_model}")
    print(f"  图神经网络: {'是' if args.graph_model else '否'}")
    print(f"  节点注意力: {'是' if args.nodal_attention else '否'}")
    print(f"  过去窗口: {args.windowp}")
    print(f"  未来窗口: {args.windowf}")
    print("=" * 80)
    
    print("特征提取配置 (与gpu_processor_enhanced.py一致):")
    print(f"  特征维度: {args.feature_dim}")
    print(f"  池化策略: {args.pooling_strategy}")
    print(f"  上下文窗口: {'是' if args.use_context_window else '否'}")
    if args.use_context_window:
        print(f"  窗口大小: {args.context_window_size}")
    print(f"  BERT最大长度: {args.max_length}")
    print("=" * 80)

    print("情绪标签映射 (4类):")
    print("  neutral (0): neutral, astonished, surprised")
    print("  happy (1): relaxed, happy, grateful, positive-other")
    print("  sad (2): depress, fear, fearful, negative-other, sadness, sad, worried")
    print("  angry (3): anger, angry, disgust, disgusted")
    print("=" * 80)

    # 构建训练命令 - 使用train_chinese_enhanced.py
    cmd = [
        'python', 'train_chinese_enhanced.py',
        '--data_path', args.data_path,
        '--output_dir', args.output_dir,
        '--epochs', str(args.n_epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--dropout', str(args.dropout),
        '--seed', str(args.seed),
        '--base-model', args.base_model,
        '--feature-dim', str(args.feature_dim),
        '--windowp', str(args.windowp),
        '--windowf', str(args.windowf),
        '--l2', str(args.l2)
    ]
    
    # 添加可选参数
    if args.graph_model:
        cmd.append('--graph-model')
    if args.nodal_attention:
        cmd.append('--nodal-attention')
    if args.use_context_window:
        cmd.append('--use-context-window')
        cmd.extend(['--context-window-size', str(args.context_window_size)])
    if args.class_weight:
        cmd.append('--class-weight')
    if args.cuda:
        cmd.append('--cuda')
    if args.use_recommended_config:
        cmd.append('--use-recommended-config')
    
    # 添加特征提取参数
    cmd.extend(['--pooling-strategy', args.pooling_strategy])
    cmd.extend(['--max-length', str(args.max_length)])

    print(f"执行命令: {' '.join(cmd)}")
    print("=" * 80)

    # 运行训练
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✓ 训练完成！")
        print(f"模型已保存到: {args.output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 训练失败: {e}")
        return
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        return

if __name__ == "__main__":
    main()