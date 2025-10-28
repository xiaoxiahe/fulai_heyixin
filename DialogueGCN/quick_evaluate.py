#!/usr/bin/env python3
"""
快速评估脚本 - 简化版启动器
自动检测可用模型并启动评估
"""
import os
import sys
from glob import glob

def find_model_files():
    """查找可用的模型文件"""
    search_paths = [
        '*.pkl',  # 当前目录优先
        'DialogueGCN/saved/*.pkl',
        'saved/*.pkl'
    ]
    
    model_files = []
    seen = set()
    for pattern in search_paths:
        found = glob(pattern)
        for f in found:
            # 规范化路径，避免重复
            normalized = os.path.normpath(f)
            if normalized not in seen:
                seen.add(normalized)
                model_files.append(f)
    
    # 过滤，优先选择best_model
    best_models = [f for f in model_files if 'best_model_Graph_LSTM_2' in f]
    checkpoints = [f for f in model_files if 'checkpoint' in f]
    
    return best_models, checkpoints

def get_per_emotion_thresholds():
    """获取个性化情绪阈值设置"""
    print("\n基于您的预测成功时的置信度统计数据，建议的个性化阈值:")
    print("  HAPPY: 0.47 (成功率77.14%，预测成功时置信度范围0.27-0.63)")
    print("  SAD: 0.44 (成功率25%，预测成功时置信度范围0.42-0.48)")
    print("  ANGRY: 0.74 (成功率100%，预测成功时置信度范围0.27-0.99)")
    print("  NEUTRAL: 0.30 (保守设置)")
    print("\n说明:")
    print("  - 预测成功 = 模型预测结果与实际标签一致")
    print("  - 置信度范围仅包含预测成功时的数据")
    print("  - 如果预测的情绪置信度未超过阈值，将强制判断为NEUTRAL")
    
    print("\n请选择阈值设置方式:")
    print("  [1] 使用建议阈值")
    print("  [2] 自定义阈值")
    
    choice = input("请选择 [1/2，默认1]: ").strip()
    
    if choice == '2':
        print("\n请输入各情绪的阈值 (格式: 情绪:阈值，用逗号分隔)")
        print("例如: HAPPY:0.35,SAD:0.40,ANGRY:0.27")
        print("注意: 阈值设置后，未达到阈值的预测将强制改为NEUTRAL")
        custom_input = input("自定义阈值: ").strip()
        
        if custom_input:
            try:
                per_emotion_thresholds = {}
                for pair in custom_input.split(','):
                    emotion, threshold = pair.strip().split(':')
                    per_emotion_thresholds[emotion.strip().upper()] = float(threshold.strip())
                return per_emotion_thresholds
            except Exception as e:
                print(f"解析失败: {e}，使用建议阈值")
    
    # 默认使用建议阈值
    return {
        'HAPPY': 0.47,
        'SAD': 0.44,
        'ANGRY': 0.74,
        'NEUTRAL': 0.30
    }

def main():
    print("="*60)
    print("DialogueGCN模型快速评估")
    print("="*60)
    
    # 查找模型
    best_models, checkpoints = find_model_files()
    
    if not best_models and not checkpoints:
        print("未找到任何模型文件（.pkl）")
        print("\n请确保已完成训练，或将模型文件放在以下位置之一:")
        print("  - DialogueGCN/saved/")
        print("  - saved/")
        print("  - 当前目录")
        return
    
    # 显示找到的模型
    print(f"\n找到 {len(best_models)} 个最佳模型, {len(checkpoints)} 个检查点")
    
    all_models = best_models + checkpoints
    print("\n可用模型:")
    for i, model_path in enumerate(all_models[:10], 1):  # 最多显示10个
        size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        mtime = os.path.getmtime(model_path)
        import time
        time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
        print(f"  [{i}] {model_path} ({size:.1f}MB, {time_str})")
    
    # 选择模型
    if len(all_models) == 1:
        selected_model = all_models[0]
        print(f"\n自动选择唯一的模型: {selected_model}")
    else:
        # 默认选择第一个best_model
        selected_model = all_models[0]
        print(f"\n默认选择: {selected_model}")
        print("(可以修改 quick_evaluate.py 中的选择逻辑)")
    
    # 选择评估模式
    print("\n评估模式:")
    print("  [1] 交互式评估 (推荐)")
    print("  [2] 文件批量评估")
    print("  [3] 交互式评估 (置信度阈值)")
    print("  [4] 文件批量评估 (置信度阈值)")
    print("  [5] 交互式评估 (个性化阈值)")
    print("  [6] 文件批量评估 (个性化阈值)")
    
    mode_input = input("\n请选择模式 [1/2/3/4/5/6，默认1]: ").strip()
    
    if mode_input == '2':
        mode = 'file'
        test_file = input("请输入测试文件路径 [默认: improved_test_data.xlsx]: ").strip()
        if not test_file:
            test_file = 'improved_test_data.xlsx'
        confidence_threshold = None
        per_emotion_thresholds = None
    elif mode_input == '3':
        mode = 'interactive'
        test_file = None
        confidence_threshold = 0.4
        per_emotion_thresholds = None
        print("已启用置信度阈值模式：非neutral情绪需置信度>40%")
    elif mode_input == '4':
        mode = 'file'
        test_file = input("请输入测试文件路径 [默认: improved_test_data.xlsx]: ").strip()
        if not test_file:
            test_file = 'improved_test_data.xlsx'
        confidence_threshold = 0.5
        per_emotion_thresholds = None
        print("已启用置信度阈值模式：非neutral情绪需置信度>50%")
    elif mode_input == '5':
        mode = 'interactive'
        test_file = None
        confidence_threshold = None
        per_emotion_thresholds = get_per_emotion_thresholds()
    elif mode_input == '6':
        mode = 'file'
        test_file = input("请输入测试文件路径 [默认: improved_test_data.xlsx]: ").strip()
        if not test_file:
            test_file = 'improved_test_data.xlsx'
        confidence_threshold = None
        per_emotion_thresholds = get_per_emotion_thresholds()
    else:
        mode = 'interactive'
        test_file = None
        confidence_threshold = None
        per_emotion_thresholds = None
    
    # 构建命令
    cmd_parts = [
        sys.executable,
        'DialogueGCN/evaluate_trained_model.py',
        '--model_path', selected_model,
        '--mode', mode,
        '--device', 'auto'
    ]
    
    if test_file:
        cmd_parts.extend(['--test_file', test_file])
    
    if confidence_threshold is not None:
        cmd_parts.extend(['--confidence_threshold', str(confidence_threshold)])
    
    if per_emotion_thresholds is not None:
        # 将字典转换为字符串格式
        threshold_str = ','.join([f"{emotion}:{threshold}" for emotion, threshold in per_emotion_thresholds.items()])
        cmd_parts.extend(['--per_emotion_thresholds', threshold_str])
    
    cmd = ' '.join(cmd_parts)
    
    print("\n" + "="*60)
    print("执行增强版DialogueGCN评估")
    print("="*60)
    print("新功能:")
    print("  各类情绪详细统计 (精确率、召回率、F1分数)")
    print("  混淆矩阵分析")
    print("  详细性能统计")
    print("  错误分析")
    print("  与训练时一致的特征提取")
    if confidence_threshold is not None:
        print(f"  全局置信度阈值过滤 (非neutral需>{confidence_threshold*100:.0f}%)")
    if per_emotion_thresholds is not None:
        print("  个性化情绪阈值过滤:")
        for emotion, threshold in per_emotion_thresholds.items():
            print(f"    {emotion}: {threshold*100:.0f}%")
    print("="*60)
    print("执行命令:")
    print(cmd)
    print("="*60)
    print()
    
    # 执行
    import subprocess
    try:
        subprocess.run(cmd_parts)
    except KeyboardInterrupt:
        print("\n\n评估已取消")
    except Exception as e:
        print(f"\n执行失败: {e}")
        print("\n你可以手动运行以下命令:")
        print(cmd)

if __name__ == "__main__":
    main()

