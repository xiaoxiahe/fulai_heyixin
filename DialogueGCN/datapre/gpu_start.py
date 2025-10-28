# GPU加速启动脚本
# 使用GPU处理CPED数据集，支持13种情绪标签映射

import os
import sys
import time

def gpu_process(input_file, output_file=None):
    """
    GPU加速处理CPED数据
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出PKL文件路径（可选）
    """
    # 如果没有指定输出文件，使用默认名称
    if output_file is None:
        input_dir = os.path.dirname(input_file)
        input_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(input_dir, f"{input_name}_gpu_features.pkl")
    
    print("=" * 60)
    print("GPU加速处理CPED数据")
    print("=" * 60)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件不存在 {input_file}")
        return False
    
    # 检查文件大小
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"文件大小: {file_size_mb:.1f} MB")
    
    # 检查GPU
    import torch
    if torch.cuda.is_available():
        print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        estimated_minutes = file_size_mb * 0.1  # GPU模式每MB约0.1分钟
    else:
        print("⚠️ 未检测到GPU，将使用CPU模式")
        estimated_minutes = file_size_mb * 0.5
    
    print(f"估算处理时间: {estimated_minutes:.1f} 分钟")
    
    start_time = time.time()
    
    try:
        # 导入GPU处理器
        from gpu_processor import GPUProcessor
        
        # 初始化处理器
        print("正在初始化GPU处理器...")
        processor = GPUProcessor(
            batch_size=64,  # GPU可以使用更大的批处理
        )
        
        # 处理数据
        print("开始GPU加速处理数据...")
        data = processor.process_cped_data(input_file, output_file)
        
        end_time = time.time()
        processing_time = (end_time - start_time) / 60
        
        print("✅ 数据处理完成！")
        print(f"输出文件: {output_file}")
        print(f"说话者映射: {output_file.replace('.pkl', '_speaker_map.pkl')}")
        print(f"实际处理时间: {processing_time:.1f} 分钟")
        
        # 显示情绪标签映射信息
        print("\n📊 情绪标签映射:")
        print("relaxed, happy, grateful, positive-other → happy (1)")
        print("neutral → neutral (0)")
        print("astonished → surprise (5)")
        print("depress, fear, negative-other, sadness, worried → sad (2)")
        print("anger, disgust → angry (3)")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        print("\n💡 建议:")
        print("1. 检查GPU驱动和CUDA版本")
        print("2. 确保有足够的GPU内存")
        print("3. 尝试减小批处理大小")
        return False

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("python gpu_start.py <输入文件> [输出文件]")
        print("")
        print("示例:")
        print("python gpu_start.py test_split.csv")
        print("python gpu_start.py test_split.csv my_gpu_features.pkl")
        print("")
        print("GPU加速特性:")
        print("- 使用GPU加速BERT特征提取")
        print("- 支持13种CPED情绪标签映射")
        print("- 批处理大小: 64")
        print("- 处理速度: 比CPU快10-50倍")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = gpu_process(input_file, output_file)
    
    if success:
        print("\n🎉 GPU加速处理成功！")
        print("现在可以开始训练DialogueGCN模型了。")
    else:
        print("\n💥 处理失败！")
        print("请检查GPU环境和系统资源。")

if __name__ == "__main__":
    main()
