#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文多轮对话情绪识别训练脚本
基于DialogueGCN模型
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
from torch.utils.data.sampler import SubsetRandomSampler
from chinese_dataloader import ChineseDataset, ChineseDatasetWithBERT
from model import MaskedNLLLoss, LSTMModel, GRUModel, DialogRNNModel, DialogueGCNModel
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
import os
import json

# 设置随机种子
seed = 100

def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_chinese_loaders(pkl_path, batch_size=32, num_workers=0, pin_memory=False, use_bert=False):
    """
    获取中文数据集的数据加载器
    
    Args:
        pkl_path: pkl文件路径
        batch_size: 批处理大小
        num_workers: 数据加载器工作进程数
        pin_memory: 是否使用pin_memory
        use_bert: 是否使用BERT特征提取
    """
    if use_bert:
        trainset = ChineseDatasetWithBERT(pkl_path, 'train')
        validset = ChineseDatasetWithBERT(pkl_path, 'valid')
        testset = ChineseDatasetWithBERT(pkl_path, 'test')
    else:
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


def train_or_eval_model(model, loss_function, dataloader, epoch, cuda, optimizer=None, train=False):
    """训练或评估模型"""
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer is not None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        # 解包数据
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        max_sequence_len.append(textf.size(0))
        
        # 前向传播
        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)  # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]


def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, optimizer=None, train=False):
    """训练或评估图模型"""
    losses, preds, labels = [], [], []
    scores, vids = [], []

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
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el


def save_results(results, output_dir):
    """保存训练结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练历史
    with open(os.path.join(output_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"训练结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='中文多轮对话情绪识别训练')
    
    # 数据相关参数
    parser.add_argument('--data_path', required=True, help='pkl数据文件路径')
    parser.add_argument('--output_dir', default='./saved/chinese/', help='模型保存目录')
    parser.add_argument('--use_bert', action='store_true', help='是否使用BERT特征提取')
    
    # 模型相关参数
    parser.add_argument('--no-cuda', action='store_true', default=False, help='不使用GPU')
    parser.add_argument('--base-model', default='LSTM', help='基础循环模型 (DialogRNN/LSTM/GRU)')
    parser.add_argument('--graph-model', action='store_true', default=False, help='是否使用图模型')
    parser.add_argument('--nodal-attention', action='store_true', default=False, help='是否使用节点注意力')
    parser.add_argument('--windowp', type=int, default=10, help='过去窗口大小')
    parser.add_argument('--windowf', type=int, default=10, help='未来窗口大小')
    
    # 训练相关参数
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--l2', type=float, default=0.00001, help='L2正则化权重')
    parser.add_argument('--rec-dropout', type=float, default=0.1, help='循环层dropout率')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout率')
    parser.add_argument('--batch-size', type=int, default=32, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=60, help='训练轮数')
    parser.add_argument('--class-weight', action='store_true', default=False, help='使用类别权重')
    parser.add_argument('--active-listener', action='store_true', default=False, help='主动监听器')
    parser.add_argument('--attention', default='general', help='注意力类型')
    
    args = parser.parse_args()
    print("训练参数:", args)

    # 检查CUDA
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('使用GPU训练')
    else:
        print('使用CPU训练')

    # 检查数据文件
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        return

    # 设置随机种子
    seed_everything()

    # 模型参数
    n_classes = 6  # 中文数据集有6种情绪
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size

    # 根据是否使用BERT调整特征维度
    if args.use_bert:
        D_m = 768  # BERT特征维度
    else:
        D_m = 100  # 默认特征维度

    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100

    # 创建模型
    if args.graph_model:
        model = DialogueGCNModel(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=4,  # 支持最多4个说话者
                                 max_seq_len=200,  # 增加最大序列长度
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda)
        print(f'使用图神经网络，基础模型: {args.base_model}')
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
        
        print(f'使用基础模型: {args.base_model}')
        name = 'Base'

    if cuda:
        model.cuda()

    # 设置损失函数
    if args.class_weight:
        # 中文数据集的类别权重 (需要根据实际数据调整)
        loss_weights = torch.FloatTensor([1/0.3, 1/0.2, 1/0.15, 1/0.15, 1/0.1, 1/0.1])
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

    # 获取数据加载器
    train_loader, valid_loader, test_loader = get_chinese_loaders(
        args.data_path, 
        batch_size=batch_size, 
        num_workers=0,
        use_bert=args.use_bert
    )

    # 训练循环
    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    training_history = {
        'train_loss': [], 'train_acc': [], 'train_fscore': [],
        'valid_loss': [], 'valid_acc': [], 'valid_fscore': [],
        'test_loss': [], 'test_acc': [], 'test_fscore': []
    }

    print("开始训练...")
    for e in range(n_epochs):
        start_time = time.time()

        if args.graph_model:
            train_loss, train_acc, _, _, train_fscore, _, _, _, _, _ = train_or_eval_graph_model(
                model, loss_function, train_loader, e, cuda, optimizer, True)
            valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _ = train_or_eval_graph_model(
                model, loss_function, valid_loader, e, cuda)
            test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(
                model, loss_function, test_loader, e, cuda)
        else:
            train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(
                model, loss_function, train_loader, e, optimizer, True)
            valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(
                model, loss_function, valid_loader, e)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(
                model, loss_function, test_loader, e)

        # 记录训练历史
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['train_fscore'].append(train_fscore)
        training_history['valid_loss'].append(valid_loss)
        training_history['valid_acc'].append(valid_acc)
        training_history['valid_fscore'].append(valid_fscore)
        training_history['test_loss'].append(test_loss)
        training_history['test_acc'].append(test_acc)
        training_history['test_fscore'].append(test_fscore)

        all_fscore.append(test_fscore)
        all_acc.append(test_acc)
        all_loss.append(test_loss)

        # 保存最佳模型
        if best_fscore is None or test_fscore > best_fscore:
            best_fscore = test_fscore
            best_loss = test_loss
            best_label = test_label
            best_pred = test_pred
            best_mask = test_mask if not args.graph_model else None
            
            # 保存模型
            model_path = os.path.join(args.output_dir, f'best_model_{name}_{args.base_model}.pkl')
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': e,
                'test_fscore': test_fscore,
                'test_acc': test_acc,
                'args': args
            }, model_path)

        print(f'轮次: {e+1}, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%, 训练F1: {train_fscore:.2f}%')
        print(f'        验证损失: {valid_loss:.4f}, 验证准确率: {valid_acc:.2f}%, 验证F1: {valid_fscore:.2f}%')
        print(f'        测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%, 测试F1: {test_fscore:.2f}%')
        print(f'        时间: {time.time()-start_time:.2f}秒\n')

    # 保存训练结果
    save_results(training_history, args.output_dir)

    # 打印最终结果
    print('=== 训练完成 ===')
    print(f'最佳测试F1分数: {max(all_fscore):.2f}%')
    print(f'最佳测试准确率: {max(all_acc):.2f}%')
    print(f'最佳模型已保存到: {args.output_dir}')

    # 打印分类报告
    if best_label is not None and best_pred is not None:
        print('\n=== 分类报告 ===')
        emotion_names = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise']
        print(classification_report(best_label, best_pred, target_names=emotion_names))


if __name__ == '__main__':
    main()