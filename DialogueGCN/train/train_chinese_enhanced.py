#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文多轮对话情绪识别训练脚本 - 增强优化版
参考IEMOCAP、DailyDialog和mianzhang的最佳实践
"""

import numpy as np
import argparse
import time
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from chinese_dataloader import ChineseDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, DialogRNNModel, DialogueGCNModel
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import os
import json

seed = 100

def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def detect_feature_dimension(pkl_path):
    """自动检测特征维度"""
    print("正在自动检测特征维度...")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 检查数据格式
        if isinstance(data, tuple):
            if len(data) == 6:
                # 旧格式: (video_ids, video_speakers, video_labels, video_text, trainVids, testVids)
                video_ids, video_speakers, video_labels, video_text, trainVids, testVids = data
                devVids = []  # 没有验证集
            elif len(data) == 7:
                # 新格式: (video_ids, video_speakers, video_labels, video_text, trainVids, devVids, testVids)
                video_ids, video_speakers, video_labels, video_text, trainVids, devVids, testVids = data
            else:
                raise ValueError(f"不支持的数据格式: 元组长度 {len(data)}")
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}")
        
        first_vid = video_ids[0]
        first_features = video_text[first_vid]
        
        if len(first_features) > 0:
            feature_dim = len(first_features[0])
        else:
            raise ValueError("无法检测特征维度：数据为空")
        
        if feature_dim == 768:
            pooling_type = "单一池化 (cls/mean/max/attention)"
        elif feature_dim == 2304:
            pooling_type = "多池化组合 (multi)"
        else:
            pooling_type = f"自定义维度"
        
        print(f"✓ 检测完成:")
        print(f"  特征维度: {feature_dim}")
        print(f"  池化类型: {pooling_type}")
        print(f"  训练样本数: {len(trainVids)}")
        if devVids:
            print(f"  验证样本数: {len(devVids)}")
        print(f"  测试样本数: {len(testVids)}")
        
        return feature_dim, pooling_type
        
    except Exception as e:
        print(f"❌ 特征维度检测失败: {e}")
        print("使用默认维度768")
        return 768, "默认"


def get_chinese_loaders(pkl_path, batch_size=32, num_workers=0, pin_memory=False):
    """获取中文数据集的数据加载器"""
    trainset = ChineseDataset(pkl_path, 'train')
    validset = ChineseDataset(pkl_path, 'valid')
    testset = ChineseDataset(pkl_path, 'test')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, optimizer=None, train=False):
    """训练或评估图模型 - 参考IEMOCAP实现"""
    losses, preds, labels = [], [], []
    
    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer is not None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label)

        ei = torch.cat([ei, e_i], dim=1)
        et = torch.cat([et, e_t])
        en = torch.cat([en, e_n])
        el += e_l

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan')

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore


def compute_class_weights(train_loader, n_classes, strategy='sqrt_smooth'):
    """
    计算类别权重 - 参考IEMOCAP和DailyDialog的方法
    
    Args:
        train_loader: 训练数据加载器
        n_classes: 类别数量
        strategy: 权重策略
            - 'inverse': 简单反比例
            - 'sqrt_smooth': 平方根平滑（推荐）
            - 'effective_num': 有效样本数方法
    """
    print("\n计算类别权重...")
    
    # 收集训练集标签
    all_train_labels = []
    for data in train_loader:
        _, _, _, _, mask, labels, _ = data
        mask_np = mask.cpu().numpy()
        labels_np = labels.cpu().numpy()
        for i in range(labels_np.shape[0]):
            valid_length = int(mask_np[i].sum())
            all_train_labels.extend(labels_np[i, :valid_length])
    
    # 统计类别分布
    from collections import Counter
    class_counts = Counter(all_train_labels)
    n_samples = len(all_train_labels)
    
    print("训练集类别分布:")
    emotion_names = ['NEUTRAL', 'HAPPY', 'SAD', 'ANGRY']
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        ratio = count / n_samples * 100
        name = emotion_names[int(cls)] if int(cls) < len(emotion_names) else f"Class_{cls}"
        print(f"  {name:8s}: {count:4d} 样本 ({ratio:5.2f}%)")
    
    # 计算权重
    loss_weights_np = np.zeros(n_classes)
    
    if strategy == 'inverse':
        # 策略1: 简单反比例（类似IEMOCAP）
        for cls in range(n_classes):
            if cls in class_counts:
                loss_weights_np[cls] = n_samples / (n_classes * class_counts[cls])
            else:
                loss_weights_np[cls] = 1.0
                
    elif strategy == 'sqrt_smooth':
        # 策略2: 平方根平滑（推荐）
        for cls in range(n_classes):
            if cls in class_counts:
                loss_weights_np[cls] = np.sqrt(n_samples / class_counts[cls])
            else:
                loss_weights_np[cls] = 1.0
        # 归一化
        loss_weights_np = loss_weights_np / loss_weights_np.min()
        # 限制最大权重
        loss_weights_np = np.clip(loss_weights_np, 1.0, 2.5)
        
    elif strategy == 'effective_num':
        # 策略3: 有效样本数方法（最温和）
        beta = 0.9999
        for cls in range(n_classes):
            if cls in class_counts:
                effective_num = 1.0 - np.power(beta, class_counts[cls])
                loss_weights_np[cls] = (1.0 - beta) / effective_num
            else:
                loss_weights_np[cls] = 1.0
        loss_weights_np = loss_weights_np / loss_weights_np.min()
    
    loss_weights = torch.FloatTensor(loss_weights_np)
    
    print(f"使用权重策略: {strategy}")
    print(f"计算得到的类别权重: {loss_weights.numpy()}")
    print(f"权重比例 (最大/最小): {loss_weights.max()/loss_weights.min():.2f}x")
    
    return loss_weights


def get_recommended_config(feature_dim):
    """根据特征维度返回推荐的模型配置"""
    if feature_dim == 768:
        # 标准768维特征配置
        return {
            'D_g': 200,
            'D_p': 200,
            'D_e': 150,
            'D_h': 150,
            'D_a': 100,
            'graph_h': 150,
            'dropout': 0.35,
            'rec_dropout': 0.1,
            'lr': 0.0001,
            'batch_size': 32
        }
    elif feature_dim >= 2000:  # 2304或更大
        # 大维度特征配置
        return {
            'D_g': 256,
            'D_p': 256,
            'D_e': 200,
            'D_h': 200,
            'D_a': 150,
            'graph_h': 200,
            'dropout': 0.4,
            'rec_dropout': 0.1,
            'lr': 0.00005,
            'batch_size': 16
        }
    else:
        # 默认配置
        return {
            'D_g': 150,
            'D_p': 150,
            'D_e': 100,
            'D_h': 100,
            'D_a': 100,
            'graph_h': 100,
            'dropout': 0.4,
            'rec_dropout': 0.1,
            'lr': 0.0001,
            'batch_size': 32
        }


def save_results(results, output_dir):
    """保存训练结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"训练结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='中文多轮对话情绪识别训练（增强优化版）')
    
    # 数据相关参数
    parser.add_argument('--data_path', required=True, help='pkl数据文件路径')
    parser.add_argument('--output_dir', default='./saved/chinese_enhanced/', help='模型保存目录')
    
    # 模型相关参数
    parser.add_argument('--no-cuda', action='store_true', default=False, help='不使用GPU')
    parser.add_argument('--base-model', default='LSTM', help='基础循环模型 (DialogRNN/LSTM/GRU)')
    parser.add_argument('--graph-model', action='store_true', default=False, help='是否使用图模型')
    parser.add_argument('--nodal-attention', action='store_true', default=False, help='是否使用节点注意力')
    parser.add_argument('--windowp', type=int, default=10, help='过去窗口大小')
    parser.add_argument('--windowf', type=int, default=10, help='未来窗口大小')
    
    # 训练相关参数
    parser.add_argument('--lr', type=float, default=None, help='学习率（None则自动选择）')
    parser.add_argument('--l2', type=float, default=0.00001, help='L2正则化权重')
    parser.add_argument('--rec-dropout', type=float, default=None, help='循环层dropout率（None则自动选择）')
    parser.add_argument('--dropout', type=float, default=None, help='dropout率（None则自动选择）')
    parser.add_argument('--batch-size', type=int, default=None, help='批处理大小（None则自动选择）')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--class-weight', action='store_true', default=False, help='使用类别权重')
    parser.add_argument('--weight-strategy', default='sqrt_smooth', 
                       choices=['inverse', 'sqrt_smooth', 'effective_num'],
                       help='类别权重计算策略')
    parser.add_argument('--active-listener', action='store_true', default=False, help='主动监听器')
    parser.add_argument('--attention', default='general', help='注意力类型')
    
    # 特征维度（可选）
    parser.add_argument('--feature-dim', type=int, default=None, help='特征维度（不提供则自动检测）')
    parser.add_argument('--use-recommended-config', action='store_true', default=True,
                       help='使用推荐的模型配置')
    
    args = parser.parse_args()
    print("="*80)
    print("训练参数:", args)
    print("="*80)

    # 检查CUDA
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('✓ 使用GPU训练')
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠ 使用CPU训练')

    # 检查数据文件
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        return

    # 自动检测特征维度
    if args.feature_dim is None:
        D_m, pooling_type = detect_feature_dimension(args.data_path)
    else:
        D_m = args.feature_dim
        pooling_type = "手动指定"
        print(f"使用手动指定的特征维度: {D_m}")
    
    # 获取推荐配置
    if args.use_recommended_config:
        recommended_config = get_recommended_config(D_m)
        print("\n使用推荐配置:")
        for key, value in recommended_config.items():
            print(f"  {key}: {value}")
        
        # 应用推荐配置（如果用户没有手动指定）
        D_g = recommended_config['D_g']
        D_p = recommended_config['D_p']
        D_e = recommended_config['D_e']
        D_h = recommended_config['D_h']
        D_a = recommended_config['D_a']
        graph_h = recommended_config['graph_h']
        
        if args.dropout is None:
            args.dropout = recommended_config['dropout']
        if args.rec_dropout is None:
            args.rec_dropout = recommended_config['rec_dropout']
        if args.lr is None:
            args.lr = recommended_config['lr']
        if args.batch_size is None:
            args.batch_size = recommended_config['batch_size']
    else:
        # 使用原始配置
        D_g = 150
        D_p = 150
        D_e = 100
        D_h = 100
        D_a = 100
        graph_h = 100
        
        if args.dropout is None:
            args.dropout = 0.5
        if args.rec_dropout is None:
            args.rec_dropout = 0.1
        if args.lr is None:
            args.lr = 0.0001
        if args.batch_size is None:
            args.batch_size = 32
    
    # 设置随机种子
    seed_everything()

    # 模型参数
    n_classes = 4
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size

    print(f"\n最终模型配置:")
    print(f"  特征维度 (D_m): {D_m}")
    print(f"  全局维度 (D_g): {D_g}")
    print(f"  说话者维度 (D_p): {D_p}")
    print(f"  情绪维度 (D_e): {D_e}")
    print(f"  隐藏维度 (D_h): {D_h}")
    print(f"  注意力维度 (D_a): {D_a}")
    print(f"  图维度 (graph_h): {graph_h}")
    print(f"  学习率: {args.lr}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Rec-Dropout: {args.rec_dropout}")
    print(f"  Batch Size: {batch_size}")

    # 创建模型
    if args.graph_model:
        model = DialogueGCNModel(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=4,
                                 max_seq_len=300,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda)
        print(f'\n✓ 使用图神经网络，基础模型: {args.base_model}')
        name = 'Graph'
    else:
        if args.base_model == 'DialogRNN':
            model = DialogRNNModel(D_m, D_g, D_p, D_e, D_h, D_a, 
                                   n_classes=n_classes,
                                   listener_state=args.active_listener,
                                   context_attention=args.attention,
                                   dropout_rec=args.rec_dropout,
                                   dropout=args.dropout)
        elif args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)
        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)
        else:
            print('基础模型必须是 DialogRNN/LSTM/GRU 之一')
            raise NotImplementedError
        
        print(f'\n✓ 使用基础模型: {args.base_model}')
        name = 'Base'

    if cuda:
        model.cuda()
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 获取数据加载器
    try:
        train_loader, valid_loader, test_loader = get_chinese_loaders(
            args.data_path, 
            batch_size=batch_size, 
            num_workers=0
        )
        print(f"\n✓ 数据加载器创建成功")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 设置损失函数
    if args.class_weight:
        loss_weights = compute_class_weights(train_loader, n_classes, strategy=args.weight_strategy)
        
        if args.graph_model:
            loss_function = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        if args.graph_model:
            loss_function = nn.NLLLoss()
        else:
            loss_function = MaskedNLLLoss()

    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    
    # 学习率调度器（移除verbose参数以兼容新版PyTorch）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    print("✓ 学习率调度器已初始化 (ReduceLROnPlateau)")

    # 训练循环
    best_fscore, best_loss, best_label, best_pred = None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    training_history = {
        'train_loss': [], 'train_acc': [], 'train_fscore': [],
        'valid_loss': [], 'valid_acc': [], 'valid_fscore': [],
        'test_loss': [], 'test_acc': [], 'test_fscore': [],
        'config': {
            'feature_dim': D_m,
            'pooling_type': pooling_type,
            'base_model': args.base_model,
            'graph_model': args.graph_model,
            'learning_rate': args.lr,
            'dropout': args.dropout,
            'batch_size': batch_size,
            'weight_strategy': args.weight_strategy if args.class_weight else 'none'
        }
    }

    print("\n" + "="*80)
    print("开始训练...")
    print("="*80)
    
    for e in range(n_epochs):
        start_time = time.time()

        if args.graph_model:
            train_loss, train_acc, _, _, train_fscore = train_or_eval_graph_model(
                model, loss_function, train_loader, e, cuda, optimizer, True)
            valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_graph_model(
                model, loss_function, valid_loader, e, cuda)
            test_loss, test_acc, test_label, test_pred, test_fscore = train_or_eval_graph_model(
                model, loss_function, test_loader, e, cuda)
        else:
            print("请使用--graph-model参数")
            return

        # 学习率调度
        scheduler.step(valid_fscore)

        # 记录训练历史
        training_history['train_loss'].append(float(train_loss))
        training_history['train_acc'].append(float(train_acc))
        training_history['train_fscore'].append(float(train_fscore))
        training_history['valid_loss'].append(float(valid_loss))
        training_history['valid_acc'].append(float(valid_acc))
        training_history['valid_fscore'].append(float(valid_fscore))
        training_history['test_loss'].append(float(test_loss))
        training_history['test_acc'].append(float(test_acc))
        training_history['test_fscore'].append(float(test_fscore))

        all_fscore.append(test_fscore)
        all_acc.append(test_acc)
        all_loss.append(test_loss)

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 每个epoch保存一个checkpoint
        epoch_checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{e+1:03d}_{name}_{args.base_model}.pkl')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': e + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_fscore': train_fscore,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
            'valid_fscore': valid_fscore,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_fscore': test_fscore,
            'feature_dim': D_m,
            'pooling_type': pooling_type,
            'config': {
                'D_m': D_m,
                'D_e': D_e,
                'D_h': D_h,
                'D_g': D_g if args.graph_model else None,
                'D_p': D_p if args.base_model == 'DialogRNN' else None,
                'D_a': D_a if args.base_model == 'DialogRNN' else None,
                'graph_h': graph_h if args.graph_model else None
            },
            'args': args,
            'training_history': training_history
        }, epoch_checkpoint_path)
        print(f'  [Checkpoint] Epoch {e+1} 已保存: {os.path.basename(epoch_checkpoint_path)}')
        
        # 保存最佳模型
        if best_fscore is None or test_fscore > best_fscore:
            best_fscore = test_fscore
            best_loss = test_loss
            best_label = test_label
            best_pred = test_pred
            
            best_model_path = os.path.join(args.output_dir, f'best_model_{name}_{args.base_model}.pkl')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': e + 1,
                'test_fscore': test_fscore,
                'test_acc': test_acc,
                'test_loss': test_loss,
                'train_fscore': train_fscore,
                'train_acc': train_acc,
                'valid_fscore': valid_fscore,
                'valid_acc': valid_acc,
                'feature_dim': D_m,
                'pooling_type': pooling_type,
                'config': {
                    'D_m': D_m,
                    'D_e': D_e,
                    'D_h': D_h,
                    'D_g': D_g if args.graph_model else None,
                    'D_p': D_p if args.base_model == 'DialogRNN' else None,
                    'D_a': D_a if args.base_model == 'DialogRNN' else None,
                    'graph_h': graph_h if args.graph_model else None
                },
                'args': args,
                'training_history': training_history
            }, best_model_path)
            
            print(f'  [最佳模型] 已保存: {os.path.basename(best_model_path)}')

        # 打印进度
        print(f'Epoch {e+1:02d}/{n_epochs}:')
        print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:5.2f}%, F1: {train_fscore:5.2f}%')
        print(f'  Valid - Loss: {valid_loss:.4f}, Acc: {valid_acc:5.2f}%, F1: {valid_fscore:5.2f}%')
        print(f'  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:5.2f}%, F1: {test_fscore:5.2f}% {"[BEST]" if test_fscore == best_fscore else ""}')
        print(f'  Time: {time.time()-start_time:.2f}s\n')

    # 保存训练结果
    save_results(training_history, args.output_dir)

    # 打印最终结果
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    print(f'最佳测试F1分数: {max(all_fscore):.2f}%')
    print(f'最佳测试准确率: {max(all_acc):.2f}%')
    print(f'最佳模型已保存到: {args.output_dir}')

    # 打印分类报告
    if best_label is not None and best_pred is not None:
        print('\n=== 分类报告 ===')
        emotion_names = ['neutral', 'happy', 'sad', 'angry']
        print(classification_report(best_label, best_pred, target_names=emotion_names))
        
        print('\n=== 混淆矩阵 ===')
        cm = confusion_matrix(best_label, best_pred)
        print(cm)


if __name__ == '__main__':
    main()

