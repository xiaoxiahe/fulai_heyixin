#!/usr/bin/env python3
"""
使用训练好的DialogueGCN模型进行评估
参考final_emotion_system.py的逻辑
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

# 导入BERT相关库
try:
    from transformers import AutoTokenizer, AutoModel
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️ BERT库未安装，将使用简化特征提取")

# 导入模型
try:
    from model import DialogueGCNModel
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("❌ 无法导入DialogueGCN模型")

class DialogueGCNEvaluator:
    """DialogueGCN模型评估器"""
    
    def __init__(self, model_path, device='auto', confidence_threshold=None, per_emotion_thresholds=None):
        """初始化评估器
        
        Args:
            model_path: 模型文件路径（.pkl文件）
            device: 设备 ('cuda', 'cpu', 或 'auto')
            confidence_threshold: 全局置信度阈值，非neutral情绪需超过此阈值才被接受
            per_emotion_thresholds: 每类情绪的个性化阈值字典，格式：{'HAPPY': 0.35, 'SAD': 0.40, 'ANGRY': 0.27}
        """
        # 自动检测设备
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                print("🚀 检测到CUDA，使用GPU加速")
            else:
                self.device = 'cpu'
                print("💻 使用CPU运行")
        else:
            self.device = device
        
        # 情绪标签映射（与训练时一致 - train_chinese_auto_dim.py中n_classes=4）
        self.emotion_map = {
            0: 'NEUTRAL',
            1: 'HAPPY',
            2: 'SAD',
            3: 'ANGRY'
        }
        
        self.reverse_emotion_map = {v: k for k, v in self.emotion_map.items()}
        
        # 对话历史
        self.conversation_history = []
        
        # 置信度阈值
        self.confidence_threshold = confidence_threshold
        self.per_emotion_thresholds = per_emotion_thresholds or {}
        
        if confidence_threshold is not None:
            print(f"🎚️ 全局置信度阈值已设置: {confidence_threshold:.1%} (非neutral情绪需超过此阈值)")
        
        if per_emotion_thresholds:
            print("🎚️ 个性化情绪阈值已设置:")
            for emotion, threshold in per_emotion_thresholds.items():
                print(f"   {emotion}: {threshold:.1%}")
        
        # 初始化组件
        self.model = None
        self.tokenizer = None
        self.bert_model = None
        
        # 加载模型和BERT
        self.load_model(model_path)
        self.load_bert()
    
    def reset_dialogue_context(self):
        """重置对话上下文"""
        self.conversation_history = []
        print("🔄 对话上下文已重置")
    
    def load_model(self, model_path):
        """加载训练好的DialogueGCN模型"""
        print(f"正在加载DialogueGCN模型: {model_path}")
        
        if not MODEL_AVAILABLE:
            print("❌ 无法导入DialogueGCN模型，请确保model.py存在")
            return False
        
        try:
            # 加载检查点
            if not os.path.exists(model_path):
                print(f"❌ 模型文件不存在: {model_path}")
                return False
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 获取模型参数
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("✅ 检测到标准检查点格式")
                model_state = checkpoint['model_state_dict']
                
                # 打印检查点信息
                if 'epoch' in checkpoint:
                    print(f"   训练Epoch: {checkpoint['epoch'] + 1}")
                if 'test_fscore' in checkpoint:
                    print(f"   测试F1: {checkpoint['test_fscore']:.2f}%")
                if 'test_acc' in checkpoint:
                    print(f"   测试准确率: {checkpoint['test_acc']:.2f}%")
                if 'feature_dim' in checkpoint:
                    feature_dim = checkpoint['feature_dim']
                    print(f"   特征维度: {feature_dim}")
                else:
                    feature_dim = 768  # 默认值
                
                # 保存特征维度到实例属性
                self.feature_dim = feature_dim
                
                # 获取训练参数和模型配置
                args = checkpoint.get('args', None)
                config = checkpoint.get('config', {})
                
                # 从检查点或配置中读取模型维度参数
                # 优先使用config，其次使用checkpoint中直接保存的值，最后使用默认值
                D_m = feature_dim
                
                # 从模型权重推断维度（更准确）
                # 注意：在DialogueGCNModel中，LSTM/GRU使用的是 hidden_size=D_e（不是D_h）
                try:
                    # 从LSTM/GRU权重推断D_e
                    if 'lstm.weight_ih_l0' in model_state:
                        lstm_weight_shape = model_state['lstm.weight_ih_l0'].shape
                        D_e = lstm_weight_shape[0] // 4  # LSTM有4个门
                        print(f"   从LSTM权重推断D_e: {D_e}")
                    elif 'gru.weight_ih_l0' in model_state:
                        gru_weight_shape = model_state['gru.weight_ih_l0'].shape
                        D_e = gru_weight_shape[0] // 3  # GRU有3个门
                        print(f"   从GRU权重推断D_e: {D_e}")
                    else:
                        D_e = config.get('D_e', 150 if D_m > 1000 else 100)
                        print(f"   使用配置或默认D_e: {D_e}")
                    
                    # 从图网络权重推断graph_h
                    if 'graph_net.conv1.bias' in model_state:
                        graph_h = model_state['graph_net.conv1.bias'].shape[0]
                        print(f"   从图网络权重推断graph_h: {graph_h}")
                    else:
                        graph_h = config.get('graph_h', 150 if D_m > 1000 else 100)
                        print(f"   使用配置或默认graph_h: {graph_h}")
                    
                    # 其他维度
                    D_g = config.get('D_g', graph_h)
                    D_p = config.get('D_p', D_g)
                    D_h = config.get('D_h', D_e)  # D_h通常等于D_e
                    D_a = config.get('D_a', 100)
                    
                except Exception as e:
                    print(f"   ⚠️ 从权重推断维度失败: {e}, 使用默认值")
                    # 根据特征维度推断其他维度（回退方案）
                    if D_m > 1000:
                        D_g = config.get('D_g', 200)
                        D_p = config.get('D_p', 200)
                        D_e = config.get('D_e', 150)
                        D_h = config.get('D_h', 150)
                        D_a = config.get('D_a', 100)
                        graph_h = config.get('graph_h', 150)
                    else:
                        D_g = config.get('D_g', 150)
                        D_p = config.get('D_p', 150)
                        D_e = config.get('D_e', 100)
                        D_h = config.get('D_h', 100)
                        D_a = config.get('D_a', 100)
                        graph_h = config.get('graph_h', 100)
                
                print(f"   模型配置: D_e={D_e}, D_h={D_h}, D_g={D_g}, graph_h={graph_h}")
                
                # 从模型权重推断base_model类型
                if 'lstm.weight_ih_l0' in model_state:
                    base_model = 'LSTM'
                elif 'gru.weight_ih_l0' in model_state:
                    base_model = 'GRU'
                elif 'dialog_rnn_f.g_cell.weight_ih' in model_state:
                    base_model = 'DialogRNN'
                else:
                    base_model = 'LSTM'  # 默认
                print(f"   检测到base_model: {base_model}")
                
                # 从checkpoint中读取训练时的窗口参数
                window_past = 10  # 默认值
                window_future = 10  # 默认值
                listener_state = False
                context_attention = 'simple'
                dropout_rec = 0.1
                dropout = 0.5
                nodal_attention = False
                avec = False
                
                # 尝试从args中读取训练时的参数
                if args:
                    window_past = getattr(args, 'windowp', 10)
                    window_future = getattr(args, 'windowf', 10)
                    listener_state = getattr(args, 'active_listener', False)
                    context_attention = getattr(args, 'attention', 'simple')
                    dropout_rec = getattr(args, 'rec_dropout', 0.1)
                    dropout = getattr(args, 'dropout', 0.5)
                    nodal_attention = getattr(args, 'nodal_attention', False)
                    avec = getattr(args, 'avec', False)
                
                print(f"   训练参数: base_model={base_model}, window_past={window_past}, window_future={window_future}")
                print(f"   注意力参数: listener_state={listener_state}, context_attention={context_attention}, nodal_attention={nodal_attention}")
                print(f"   Dropout参数: dropout={dropout}, dropout_rec={dropout_rec}")
                
                # 保存窗口参数供预测时使用
                self.window_past = window_past
                self.window_future = window_future
                
                # 从checkpoint中读取特征提取配置
                self.use_context_window = getattr(args, 'use_context_window', False)
                self.context_window_size = getattr(args, 'context_window_size', 8)
                self.pooling_strategy = getattr(args, 'pooling_strategy', 'auto')
                
                print(f"   特征提取配置: use_context_window={self.use_context_window}, context_window_size={self.context_window_size}")
                print(f"   池化策略: {self.pooling_strategy}")
                
                # 创建模型实例（与训练时一致）
                self.model = DialogueGCNModel(
                    base_model=base_model,
                    D_m=D_m,
                    D_g=D_g,
                    D_p=D_p,
                    D_e=D_e,
                    D_h=D_h,
                    D_a=D_a,
                    graph_hidden_size=graph_h,
                    n_speakers=4,
                    max_seq_len=300,
                    window_past=window_past,
                    window_future=window_future,
                    n_classes=4,
                    listener_state=listener_state,
                    context_attention=context_attention,
                    dropout_rec=dropout_rec,
                    dropout=dropout,
                    nodal_attention=nodal_attention,
                    avec=avec,
                    no_cuda=(self.device == 'cpu')
                )
                
                # 加载权重
                self.model.load_state_dict(model_state)
                print(f"✅ 成功加载模型权重")
                
            else:
                print("❌ 不支持的检查点格式")
                return False
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ 模型已设置为评估模式")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_bert(self):
        """加载BERT模型"""
        if not BERT_AVAILABLE:
            print("⚠️ transformers库不可用，将使用简化特征提取")
            return False
        
        print("正在加载chinese-roberta-wwm-ext模型...")
        try:
            cache_dir = "./bert_cache"
            os.makedirs(cache_dir, exist_ok=True)
            model_name = 'hfl/chinese-roberta-wwm-ext'
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
            self.bert_model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
            self.bert_model.to(self.device)
            self.bert_model.eval()
            print("✅ BERT模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ BERT模型加载失败: {e}")
            return False
    
    def extract_bert_features(self, text, pooling_strategy='auto'):
        """
        提取BERT特征 - 支持gpu_processor_enhanced.py的多种池化策略
        
        Args:
            text: 输入文本
            pooling_strategy: 池化策略 ('auto', 'cls', 'mean', 'max', 'multi')
        """
        if self.bert_model is None or self.tokenizer is None:
            # 如果BERT不可用，返回零向量
            return torch.zeros(self.feature_dim, device=self.device)
        
        try:
            text = text.strip()
            if not text:
                return torch.zeros(self.feature_dim, device=self.device)
            
            # 根据特征维度自动选择max_length
            if self.feature_dim == 768:
                max_length = 128  # 与preprocess_chinese_data.py一致
            elif self.feature_dim == 2304:
                max_length = 256  # 与gpu_processor_enhanced.py一致
            else:
                max_length = 128  # 默认值
            
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, 768]
                attention_mask = inputs['attention_mask']  # [batch_size, seq_len]
                
                # 根据特征维度和池化策略选择提取方式
                if self.feature_dim == 768:
                    # 单一池化：只使用[CLS]标记（与preprocess_chinese_data.py一致）
                    features = hidden_states[:, 0, :].squeeze(0)  # [768]
                elif self.feature_dim == 2304:
                    # 多池化组合：CLS + Mean + Max（与gpu_processor_enhanced.py一致）
                    features = self._multi_pooling(hidden_states, attention_mask).squeeze(0)
                else:
                    # 默认使用CLS
                    features = hidden_states[:, 0, :].squeeze(0)
                
                return features
                
        except Exception as e:
            print(f"⚠️ BERT特征提取失败: {e}")
            return torch.zeros(self.feature_dim, device=self.device)
    
    def _cls_pooling(self, hidden_states, attention_mask):
        """[CLS] token pooling"""
        return hidden_states[:, 0, :]
    
    def _mean_pooling(self, hidden_states, attention_mask):
        """均值池化"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def _max_pooling(self, hidden_states, attention_mask):
        """最大池化"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        hidden_states_clone = hidden_states.clone()
        hidden_states_clone[input_mask_expanded == 0] = -1e9
        return torch.max(hidden_states_clone, 1)[0]
    
    def _multi_pooling(self, hidden_states, attention_mask):
        """多种池化策略组合（与gpu_processor_enhanced.py一致）"""
        cls_pooled = self._cls_pooling(hidden_states, attention_mask)
        mean_pooled = self._mean_pooling(hidden_states, attention_mask)
        max_pooled = self._max_pooling(hidden_states, attention_mask)
        return torch.cat([cls_pooled, mean_pooled, max_pooled], dim=-1)
    
    def extract_contextual_features(self, texts, speakers, window_size=8):
        """
        提取带上下文信息的特征（参考gpu_processor_enhanced.py）
        
        Args:
            texts: 文本列表
            speakers: 说话者列表
            window_size: 上下文窗口大小
        
        Returns:
            增强后的特征列表
        """
        if len(texts) == 0:
            return []
        
        enhanced_features = []
        
        for idx in range(len(texts)):
            # 获取上下文窗口
            start_idx = max(0, idx - window_size)
            end_idx = min(len(texts), idx + window_size + 1)
            
            # 构建上下文
            context_texts = []
            for i in range(start_idx, end_idx):
                speaker = speakers[i] if i < len(speakers) else "?"
                text = texts[i]
                context_texts.append(f"{speaker}: {text}")
            
            context_str = " [SEP] ".join(context_texts)
            
            # 提取当前特征
            current_features = self.extract_bert_features(texts[idx])
            
            # 提取上下文特征
            context_features = self.extract_bert_features(context_str)
            
            # 组合：当前70% + 上下文30%（与gpu_processor_enhanced.py一致）
            combined = 0.7 * current_features + 0.3 * context_features
            enhanced_features.append(combined)
        
        return enhanced_features
    
    def predict_emotion(self, user_text, robot_text=""):
        """预测情绪
        
        Args:
            user_text: 用户输入文本
            robot_text: 机器人回复文本（可选）
        
        Returns:
            Dict包含emotion, confidence, probabilities
        """
        if self.model is None:
            print("❌ 模型未加载")
            return None
        
        try:
            self.model.to(self.device)
            self.model.eval()
            
            # 构建对话序列
            conversation = []
            
            # 添加历史对话（使用训练时的窗口参数）
            # 从checkpoint中获取window_past和window_future参数
            window_past = getattr(self, 'window_past', 10)  # 默认值
            window_future = getattr(self, 'window_future', 10)  # 默认值
            
            # 根据窗口参数添加历史对话
            history_length = min(len(self.conversation_history), window_past)
            for hist_text, hist_speaker in self.conversation_history[-history_length:]:
                conversation.append((hist_text, hist_speaker))
            
            # 添加当前对话
            if robot_text:
                conversation.append((robot_text, 1))  # robot
            conversation.append((user_text, 0))  # user
            
            # 限制对话长度以避免维度问题
            # 如果对话太长，只保留最近的几轮
            max_conversation_length = 10  # 限制最大对话长度
            if len(conversation) > max_conversation_length:
                conversation = conversation[-max_conversation_length:]
            
            # 提取特征 - 支持上下文窗口增强
            text_features = []
            speaker_features = []
            
            # 检查是否使用上下文窗口特征提取
            use_context_window = getattr(self, 'use_context_window', False)
            context_window_size = getattr(self, 'context_window_size', 8)
            
            if use_context_window and len(conversation) > 1:
                # 使用上下文窗口特征提取（参考gpu_processor_enhanced.py）
                texts = [text for text, _ in conversation]
                speakers = [speaker for _, speaker in conversation]
                enhanced_features = self.extract_contextual_features(texts, speakers, context_window_size)
                
                for i, (text, speaker) in enumerate(conversation):
                    if i < len(enhanced_features):
                        text_feature = enhanced_features[i]
                    else:
                        text_feature = self.extract_bert_features(text)
                    
                    if text_feature.is_cuda:
                        text_feature = text_feature.cpu()
                    text_features.append(text_feature.numpy())
                    
                    # One-hot编码说话者
                    speaker_feature = [0.0] * 4  # 4个说话者
                    speaker_feature[speaker] = 1.0
                    speaker_features.append(speaker_feature)
            else:
                # 标准特征提取
                for text, speaker in conversation:
                    text_feature = self.extract_bert_features(text)
                    if text_feature.is_cuda:
                        text_feature = text_feature.cpu()
                    text_features.append(text_feature.numpy())
                    
                    # One-hot编码说话者
                    speaker_feature = [0.0] * 4  # 4个说话者
                    speaker_feature[speaker] = 1.0
                    speaker_features.append(speaker_feature)
            
            # 转换为张量
            # 注意：DialogueGCN期望的输入格式是 (seq_len, batch_size, feature_dim)
            # 对于单个对话的预测，batch_size=1
            text_features = torch.FloatTensor(np.array(text_features)).unsqueeze(1).to(self.device)  # (seq_len, 1, feature_dim)
            speaker_features = torch.FloatTensor(np.array(speaker_features)).unsqueeze(1).to(self.device)  # (seq_len, 1, n_speakers)
            
            # 创建mask
            seq_len = text_features.shape[0]  # 第0维是seq_len
            batch_size = 1  # 单个对话预测
            
            # 确保mask的形状正确
            # MatchingAttention期望mask的形状是(batch_size, seq_len)
            umask = torch.ones(batch_size, seq_len).to(self.device)  # (batch_size, seq_len)
            qmask = speaker_features  # (seq_len, batch_size, n_speakers)
            
            # 序列长度
            lengths = [seq_len]
            
            # 预测
            with torch.no_grad():
                # 调用模型
                log_prob, _, _, _, _ = self.model(text_features, qmask, umask, lengths)
                
                # log_prob的形状应该是 (total_utterances, n_classes)
                # 获取最后一个时间步的预测
                if log_prob.dim() == 2:
                    last_logits = log_prob[-1]  # 最后一条utterance
                else:
                    last_logits = log_prob[-1, :]
                
                probabilities = torch.exp(last_logits)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
            
            # 构建概率字典
            probs = {}
            for i, emotion in self.emotion_map.items():
                probs[emotion] = probabilities[i].item()
            
            # 应用置信度阈值过滤
            final_emotion = self.emotion_map[predicted_class]
            final_confidence = confidence
            threshold_applied = False
            threshold_type = None
            
            # 对任何非NEUTRAL情绪都进行阈值检查
            if final_emotion != 'NEUTRAL':
                threshold_to_use = None
                
                # 优先使用个性化阈值
                if final_emotion in self.per_emotion_thresholds:
                    threshold_to_use = self.per_emotion_thresholds[final_emotion]
                    threshold_type = f"个性化({final_emotion})"
                # 其次使用全局阈值
                elif self.confidence_threshold is not None:
                    threshold_to_use = self.confidence_threshold
                    threshold_type = "全局"
                
                # 如果设置了阈值且置信度未达到，强制改为NEUTRAL
                if threshold_to_use is not None and confidence < threshold_to_use:
                    final_emotion = 'NEUTRAL'
                    final_confidence = probs['NEUTRAL']
                    threshold_applied = True
            
            # 更新对话历史
            if robot_text:
                self.conversation_history.append((robot_text, 1))
            self.conversation_history.append((user_text, 0))
            
            # 保持历史长度
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            result = {
                'emotion': final_emotion,
                'confidence': final_confidence,
                'probabilities': probs,
                'source': 'dialoguegcn'
            }
            
            # 如果应用了阈值过滤，添加相关信息
            if threshold_applied:
                result['original_emotion'] = self.emotion_map[predicted_class]
                result['original_confidence'] = confidence
                result['threshold_applied'] = True
                result['threshold_type'] = threshold_type
            
            return result
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_interactive(self):
        """交互式评估模式"""
        print("\n" + "="*60)
        print("🎯 DialogueGCN交互式评估")
        print("="*60)
        print("使用训练好的DialogueGCN模型进行情绪预测")
        print("支持多轮对话上下文")
        print("输入 'quit' 退出")
        print("输入 'clear' 清空对话历史")
        print("输入 'history' 查看对话历史")
        print("输入 'stats' 查看统计信息")
        print("="*60)
        
        # 统计信息
        total_predictions = 0
        prediction_times = []
        emotion_counts = Counter()
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 用户: ").strip()
                
                if user_input.lower() == 'quit':
                    if total_predictions > 0:
                        print("\n" + "="*60)
                        print("📊 会话统计")
                        print("="*60)
                        print(f"总预测次数: {total_predictions}")
                        print(f"平均耗时: {np.mean(prediction_times):.2f}ms")
                        if len(prediction_times) > 1:
                            print(f"耗时范围: {np.min(prediction_times):.2f}ms - {np.max(prediction_times):.2f}ms")
                        print(f"\n情绪分布:")
                        for emotion, count in emotion_counts.most_common():
                            print(f"  {emotion}: {count} ({count/total_predictions*100:.1f}%)")
                    print("\n👋 再见！")
                    break
                    
                elif user_input.lower() == 'clear':
                    self.reset_dialogue_context()
                    continue
                    
                elif user_input.lower() == 'history':
                    print("\n📜 对话历史:")
                    if not self.conversation_history:
                        print("   暂无对话历史")
                    else:
                        for i, (text, speaker) in enumerate(self.conversation_history[-10:]):
                            speaker_name = "用户" if speaker == 0 else "机器人"
                            print(f"   {i+1}. {speaker_name}: {text}")
                    continue
                    
                elif user_input.lower() == 'stats':
                    print("\n📊 当前统计:")
                    print(f"总预测次数: {total_predictions}")
                    if prediction_times:
                        print(f"平均耗时: {np.mean(prediction_times):.2f}ms")
                    print(f"对话历史长度: {len(self.conversation_history)}")
                    if emotion_counts:
                        print(f"情绪分布:")
                        for emotion, count in emotion_counts.most_common():
                            print(f"  {emotion}: {count} ({count/total_predictions*100:.1f}%)")
                    continue
                    
                elif not user_input:
                    print("⚠️ 请输入有效内容")
                    continue
                
                # 预测情绪
                print("\n🔍 正在使用DialogueGCN分析情绪...")
                t0 = time.perf_counter()
                result = self.predict_emotion(user_input, robot_text="")
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                
                if result:
                    emotion = result['emotion']
                    confidence = result['confidence']
                    probabilities = result['probabilities']
                    
                    # 更新统计
                    total_predictions += 1
                    prediction_times.append(elapsed_ms)
                    emotion_counts[emotion] += 1
                    
                    # 显示结果
                    print(f"\n📊 DialogueGCN预测结果:")
                    print(f"   预测情绪: {emotion} (置信度: {confidence:.2%})")
                    
                    # 如果应用了置信度阈值过滤，显示原始预测
                    if result.get('threshold_applied', False):
                        original_emotion = result['original_emotion']
                        original_confidence = result['original_confidence']
                        threshold_type = result.get('threshold_type', '未知')
                        print(f"   🎚️ {threshold_type}阈值过滤: {original_emotion} ({original_confidence:.2%}) → {emotion} ({confidence:.2%})")
                    
                    print(f"   预测耗时: {elapsed_ms:.2f}ms")
                    print(f"   所有情绪概率:")
                    for emo, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                        bar = "█" * int(prob * 20)
                        print(f"     {emo:8}: {prob:.2%} {bar}")
                else:
                    print("❌ 预测失败")
                    continue
                
                # 可选的机器人回复
                robot_input = input("\n🤖 机器人回复 (可选，直接回车跳过): ").strip()
                if robot_input:
                    self.conversation_history.append((robot_input, 1))
                    print(f"✅ 机器人回复已记录")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
                import traceback
                traceback.print_exc()
    
    def evaluate_on_file(self, data_path: str = 'improved_test_data.xlsx'):
        """使用测试文件进行批量评估（完全参考final_emotion_system.py的逻辑）
        
        Args:
            data_path: 测试数据文件路径（.xlsx或.csv）
            
        要求：
        - 数据列包含: dialogue_id, speaker(user/robot), utterance, emotion_label
        - 机器人轮作为历史写入，只有有标签的才进行预测
        - 用户轮：若无标签则仅作为历史；有标签则进行预测比对
        - 最终输出：总体准确率、各类情绪召回率、平均耗时、耗时5/50/95分位
        """
        try:
            from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
            _SK_AVAILABLE = True
        except Exception:
            _SK_AVAILABLE = False
            print("⚠️ sklearn不可用，将使用简化评估")
        
        if not os.path.exists(data_path):
            print(f"❌ 未找到测试数据文件: {data_path}")
            return
        
        try:
            if data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path)
            elif data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            else:
                print("❌ 不支持的文件格式，仅支持.xlsx或.csv")
                return
        except Exception as e:
            print(f"❌ 读取测试数据失败: {e}")
            return
        
        required_cols = {'dialogue_id', 'speaker', 'utterance', 'emotion_label'}
        if not required_cols.issubset(set(df.columns)):
            print(f"❌ 测试数据缺少必要列，需包含: {sorted(required_cols)}，当前: {list(df.columns)}")
            return
        
        # 标准化列类型
        df = df.copy()
        df['speaker'] = df['speaker'].astype(str)
        df['dialogue_id'] = df['dialogue_id']
        
        # 统计信息
        user_rows = df[(df['speaker'] == 'user') & df['emotion_label'].notna() & (df['emotion_label'].astype(str) != '')]
        robot_rows = df[(df['speaker'] == 'robot') & df['emotion_label'].notna() & (df['emotion_label'].astype(str) != '')]
        print(f"\n🧪 基于文件进行评估...")
        print(f"总记录数: {len(df)} | 用户(有标签)记录数: {len(user_rows)} | 机器人(有标签)记录数: {len(robot_rows)}")
        
        label_map = {
            'neutral': 'NEUTRAL',
            'happy': 'HAPPY',
            'sad': 'SAD',
            'angry': 'ANGRY'
        }
        valid_labels = ['NEUTRAL', 'HAPPY', 'SAD', 'ANGRY']
        
        y_true, y_pred, times = [], [], []
        # 用于统计每类情绪成功预测时的置信度
        emotion_confidence_stats = {
            'NEUTRAL': {'success_confidences': [], 'total_predictions': 0, 'success_count': 0},
            'HAPPY': {'success_confidences': [], 'total_predictions': 0, 'success_count': 0},
            'SAD': {'success_confidences': [], 'total_predictions': 0, 'success_count': 0},
            'ANGRY': {'success_confidences': [], 'total_predictions': 0, 'success_count': 0}
        }
        
        # 按对话分组：每段对话重置内部状态，复用同一模型
        for dialogue_id, group in df.groupby('dialogue_id', sort=False):
            print(f"\n=== 对话 {dialogue_id} 开始，{len(group)} 条 ===")
            # 重置内部状态，保持模型不变
            self.reset_dialogue_context()
            
            group = group.reset_index(drop=True)
            for idx, row in group.iterrows():
                speaker = str(row['speaker']).strip().lower()
                utterance = str(row['utterance']) if not pd.isna(row['utterance']) else ''
                true_label_raw = row.get('emotion_label')
                true_label = '' if pd.isna(true_label_raw) else str(true_label_raw)
                
                if speaker == 'robot':
                    # 机器人轮：先添加到历史
                    if utterance.strip():
                        self.conversation_history.append((utterance.strip(), 1))
                    
                    # 如果robot有情绪标签，也进行预测
                    if true_label.strip():
                        t0 = time.perf_counter()
                        result = self.predict_emotion(utterance, robot_text="")
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        
                        if result:
                            pred = result['emotion']
                            confidence = result['confidence']
                            probabilities = result['probabilities']
                        else:
                            pred = 'NEUTRAL'
                            confidence = 0.0
                            probabilities = {}
                            print(f"⚠️ DialogueGCN预测失败，默认为NEUTRAL")
                        
                        mapped_true = label_map.get(true_label.lower(), true_label.upper())
                        y_true.append(mapped_true)
                        y_pred.append(pred)
                        times.append(elapsed_ms)
                        
                        # 更新置信度统计
                        if mapped_true in emotion_confidence_stats:
                            emotion_confidence_stats[mapped_true]['total_predictions'] += 1
                            if pred == mapped_true:  # 预测成功
                                emotion_confidence_stats[mapped_true]['success_count'] += 1
                                # 只统计预测成功时的置信度
                                emotion_confidence_stats[mapped_true]['success_confidences'].append(confidence)
                        
                        match_symbol = "✅" if pred == mapped_true else "❌"
                        print(f"{match_symbol} Robot话语: {utterance}")
                        print(f"   耗时: {elapsed_ms:.1f}ms | 预测: {pred} (置信度: {confidence:.2%}) | 实际: {mapped_true}")
                        
                        # 如果应用了置信度阈值过滤，显示原始预测
                        if result and result.get('threshold_applied', False):
                            original_emotion = result['original_emotion']
                            original_confidence = result['original_confidence']
                            threshold_type = result.get('threshold_type', '未知')
                            print(f"   🎚️ {threshold_type}阈值过滤: {original_emotion} ({original_confidence:.2%}) → {pred} ({confidence:.2%})")
                        
                        if probabilities:
                            print(f"   各类情绪置信度: " + " | ".join([f"{emo}: {prob:.2%}" for emo, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)]))
                    continue
                
                if speaker == 'user':
                    # 用户轮：若无标签，则仅作为历史；有标签则进行预测比对
                    if not true_label.strip():
                        if utterance.strip():
                            self.conversation_history.append((utterance.strip(), 0))
                        continue
                    
                    # 有标签用户轮：执行预测
                    t0 = time.perf_counter()
                    result = self.predict_emotion(utterance, robot_text="")
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    
                    if result:
                        pred = result['emotion']
                        confidence = result['confidence']
                        probabilities = result['probabilities']
                    else:
                        pred = 'NEUTRAL'
                        confidence = 0.0
                        probabilities = {}
                        print(f"⚠️ DialogueGCN预测失败，默认为NEUTRAL")
                    
                    mapped_true = label_map.get(true_label.lower(), true_label.upper())
                    y_true.append(mapped_true)
                    y_pred.append(pred)
                    times.append(elapsed_ms)
                    
                    # 更新置信度统计
                    if mapped_true in emotion_confidence_stats:
                        emotion_confidence_stats[mapped_true]['total_predictions'] += 1
                        if pred == mapped_true:  # 预测成功
                            emotion_confidence_stats[mapped_true]['success_count'] += 1
                            # 只统计预测成功时的置信度
                            emotion_confidence_stats[mapped_true]['success_confidences'].append(confidence)
                    
                    match_symbol = "✅" if pred == mapped_true else "❌"
                    print(f"{match_symbol} User话语: {utterance}")
                    print(f"   耗时: {elapsed_ms:.1f}ms | 预测: {pred} (置信度: {confidence:.2%}) | 实际: {mapped_true}")
                    
                    # 如果应用了置信度阈值过滤，显示原始预测
                    if result and result.get('threshold_applied', False):
                        original_emotion = result['original_emotion']
                        original_confidence = result['original_confidence']
                        threshold_type = result.get('threshold_type', '未知')
                        print(f"   🎚️ {threshold_type}阈值过滤: {original_emotion} ({original_confidence:.2%}) → {pred} ({confidence:.2%})")
                    
                    if probabilities:
                        print(f"   各类情绪置信度: " + " | ".join([f"{emo}: {prob:.2%}" for emo, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)]))
        
        # 汇总指标（增强版 - 包含各类情绪的详细统计）
        if not y_true:
            print("\n❌ 没有可评估的标注数据（用户或机器人）")
            return
        
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        accuracy = float((y_true_np == y_pred_np).mean())
        
        print("\n" + "="*80)
        print("📊 DialogueGCN模型评估结果汇总")
        print("="*80)
        print(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"总样本数: {len(y_true)}")
        
        # 各类情绪的详细统计
        print("\n" + "="*80)
        print("📈 各类情绪详细统计")
        print("="*80)
        
        if _SK_AVAILABLE:
            try:
                # 计算精确率、召回率、F1分数
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true_np, y_pred_np, labels=valid_labels, zero_division=0
                )
                
                print(f"{'情绪类别':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'支持数':<8} {'准确率':<8}")
                print("-" * 70)
                
                for i, lbl in enumerate(valid_labels):
                    # 计算该类别的准确率（该类别的正确预测数 / 该类别的总预测数）
                    class_predictions = (y_pred_np == lbl).sum()
                    class_correct = ((y_true_np == lbl) & (y_pred_np == lbl)).sum()
                    class_accuracy = class_correct / class_predictions if class_predictions > 0 else 0.0
                    
                    print(f"{lbl:<12} {precision[i]:<8.4f} {recall[i]:<8.4f} {f1[i]:<8.4f} {int(support[i]):<8} {class_accuracy:<8.4f}")
                
                # 计算宏平均和微平均
                macro_precision = np.mean(precision)
                macro_recall = np.mean(recall)
                macro_f1 = np.mean(f1)
                
                # 微平均（所有类别的TP、FP、FN的总和）
                total_tp = sum(((y_true_np == lbl) & (y_pred_np == lbl)).sum() for lbl in valid_labels)
                total_fp = sum(((y_true_np != lbl) & (y_pred_np == lbl)).sum() for lbl in valid_labels)
                total_fn = sum(((y_true_np == lbl) & (y_pred_np != lbl)).sum() for lbl in valid_labels)
                
                micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
                
                print("-" * 70)
                print(f"{'宏平均':<12} {macro_precision:<8.4f} {macro_recall:<8.4f} {macro_f1:<8.4f} {'-':<8} {'-':<8}")
                print(f"{'微平均':<12} {micro_precision:<8.4f} {micro_recall:<8.4f} {micro_f1:<8.4f} {'-':<8} {'-':<8}")
                
            except Exception as e:
                print(f"  计算详细指标失败: {e}")
                # 回退到简单计算
                print("\n各类情绪召回率:")
                for lbl in valid_labels:
                    sup = int((y_true_np == lbl).sum())
                    tp = int(((y_true_np == lbl) & (y_pred_np == lbl)).sum())
                    rec = tp / sup if sup > 0 else 0.0
                    print(f"  {lbl:<9} Recall {rec:.4f} (support={sup})")
        
        # 置信度统计 - 只统计预测成功时的置信度
        print("\n" + "="*80)
        print("📊 每类情绪预测成功时的置信度统计")
        print("="*80)
        print("说明: 预测成功 = 模型预测结果与实际标签一致")
        print(f"{'情绪类别':<12} {'成功次数':<8} {'成功率':<8} {'成功时最低置信度':<16} {'成功时最高置信度':<16} {'成功时平均置信度':<16}")
        print("-" * 90)
        
        for emotion in valid_labels:
            stats = emotion_confidence_stats[emotion]
            total_pred = stats['total_predictions']
            success_count = stats['success_count']
            success_confidences = stats['success_confidences']
            
            if total_pred > 0:
                success_rate = success_count / total_pred
                if success_confidences:
                    min_conf = min(success_confidences)
                    max_conf = max(success_confidences)
                    avg_conf = sum(success_confidences) / len(success_confidences)
                    print(f"{emotion:<12} {success_count:<8} {success_rate:<8.4f} {min_conf:<16.4f} {max_conf:<16.4f} {avg_conf:<16.4f}")
                else:
                    print(f"{emotion:<12} {success_count:<8} {success_rate:<8.4f} {'N/A':<16} {'N/A':<16} {'N/A':<16}")
            else:
                print(f"{emotion:<12} {'0':<8} {'0.0000':<8} {'N/A':<16} {'N/A':<16} {'N/A':<16}")
        
        # 如果没有sklearn，使用手工计算
        if not _SK_AVAILABLE:
            print("\n" + "="*80)
            print("📈 各类情绪详细统计 (手工计算)")
            print("="*80)
            print(f"{'情绪类别':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'支持数':<8} {'准确率':<8}")
            print("-" * 70)
            
            for lbl in valid_labels:
                # 计算TP, FP, FN
                tp = int(((y_true_np == lbl) & (y_pred_np == lbl)).sum())
                fp = int(((y_true_np != lbl) & (y_pred_np == lbl)).sum())
                fn = int(((y_true_np == lbl) & (y_pred_np != lbl)).sum())
                support = int((y_true_np == lbl).sum())
                
                # 计算指标
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                accuracy = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                
                print(f"{lbl:<12} {precision:<8.4f} {recall:<8.4f} {f1:<8.4f} {support:<8} {accuracy:<8.4f}")
        
        # 耗时统计
        t = np.array(times, dtype=float)
        if t.size > 0:
            print("\n" + "="*80)
            print("⏱️ 性能统计")
            print("="*80)
            print(f"平均耗时: {t.mean():.2f}ms")
            print(f"中位数耗时: {np.median(t):.2f}ms")
            print(f"最小耗时: {t.min():.2f}ms")
            print(f"最大耗时: {t.max():.2f}ms")
            print(f"标准差: {t.std():.2f}ms")
            print(f"5分位: {np.percentile(t, 5):.2f}ms | 50分位: {np.percentile(t, 50):.2f}ms | 95分位: {np.percentile(t, 95):.2f}ms")
        else:
            print("\n⏱️ 性能统计: 无")
        
        # 混淆矩阵
        if _SK_AVAILABLE:
            try:
                cm = confusion_matrix(y_true_np, y_pred_np, labels=valid_labels)
                print("\n" + "="*80)
                print("🔍 混淆矩阵")
                print("="*80)
                print("真实\\预测  " + "  ".join([f"{l[:4]:>6}" for l in valid_labels]))
                print("-" * (10 + len(valid_labels) * 7))
                for i, lbl in enumerate(valid_labels):
                    print(f"{lbl:<10} " + "  ".join([f"{cm[i][j]:>6}" for j in range(len(valid_labels))]))
                
                # 计算每行的准确率
                print("\n各类别预测准确率:")
                for i, lbl in enumerate(valid_labels):
                    row_sum = cm[i].sum()
                    if row_sum > 0:
                        accuracy = cm[i][i] / row_sum
                        print(f"  {lbl}: {accuracy:.4f} ({cm[i][i]}/{row_sum})")
                
            except Exception as e:
                print(f"\n混淆矩阵计算失败: {e}")
        
        # 错误分析
        print("\n" + "="*80)
        print("🔍 错误分析")
        print("="*80)
        
        # 统计最常见的错误类型
        error_pairs = []
        for true_label, pred_label in zip(y_true_np, y_pred_np):
            if true_label != pred_label:
                error_pairs.append((true_label, pred_label))
        
        if error_pairs:
            from collections import Counter
            error_counter = Counter(error_pairs)
            print("最常见的错误预测:")
            for (true_lbl, pred_lbl), count in error_counter.most_common(10):
                print(f"  {true_lbl} → {pred_lbl}: {count}次")
        else:
            print("🎉 没有预测错误！")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DialogueGCN模型评估')
    parser.add_argument('--model_path', type=str, 
                       default='emotion_bias_angry.pkl',
                       help='模型文件路径')
    parser.add_argument('--mode', type=str, choices=['interactive', 'file'],
                       default='interactive',
                       help='评估模式: interactive(交互式) 或 file(文件批量评估)')
    parser.add_argument('--test_file', type=str,
                       default='improved_test_data.xlsx',
                       help='测试数据文件路径（mode=file时使用）')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                       default='auto',
                       help='运行设备')
    parser.add_argument('--confidence_threshold', type=float, default=None,
                       help='全局置信度阈值，非neutral情绪需超过此阈值才被接受，否则强制为neutral')
    parser.add_argument('--per_emotion_thresholds', type=str, default=None,
                       help='个性化情绪阈值，格式：HAPPY:0.35,SAD:0.40,ANGRY:0.27')
    
    args = parser.parse_args()
    
    # 解析个性化阈值参数
    per_emotion_thresholds = None
    if args.per_emotion_thresholds:
        try:
            per_emotion_thresholds = {}
            for pair in args.per_emotion_thresholds.split(','):
                emotion, threshold = pair.strip().split(':')
                per_emotion_thresholds[emotion.strip().upper()] = float(threshold.strip())
            print(f"解析个性化阈值: {per_emotion_thresholds}")
        except Exception as e:
            print(f"解析个性化阈值失败: {e}")
            print("请使用格式: HAPPY:0.35,SAD:0.40,ANGRY:0.27")
            return
    
    # 智能查找模型文件
    model_path = args.model_path
    if not os.path.exists(model_path):
        # 尝试在常见位置查找
        possible_paths = [
            model_path,
            os.path.join('DialogueGCN', 'saved', os.path.basename(model_path)),
            os.path.join('saved', os.path.basename(model_path)),
            os.path.join('.', os.path.basename(model_path))
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                found = True
                print(f"✅ 找到模型文件: {model_path}")
                break
        
        if not found:
            print(f"❌ 模型文件不存在: {args.model_path}")
            print("\n可用的模型文件:")
            # 搜索可能的模型文件
            search_dirs = ['.', 'DialogueGCN/saved', 'saved']
            found_models = []
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    for file in os.listdir(search_dir):
                        if file.endswith('.pkl'):
                            full_path = os.path.join(search_dir, file)
                            if full_path not in found_models:
                                found_models.append(full_path)
                                print(f"  - {full_path}")
            
            if found_models:
                print(f"\n[提示] 使用以下命令指定模型路径:")
                print(f"python DialogueGCN/evaluate_trained_model.py --model_path {found_models[0]}")
            return
    else:
        model_path = args.model_path
    
    # 创建评估器
    print(f"[模型文件] {model_path}")
    evaluator = DialogueGCNEvaluator(
        model_path, 
        device=args.device, 
        confidence_threshold=args.confidence_threshold,
        per_emotion_thresholds=per_emotion_thresholds
    )
    
    if evaluator.model is None:
        print("[错误] 模型加载失败，无法继续")
        return
    
    # 根据模式运行评估
    if args.mode == 'interactive':
        evaluator.run_interactive()
    elif args.mode == 'file':
        evaluator.evaluate_on_file(args.test_file)

if __name__ == "__main__":
    main()

