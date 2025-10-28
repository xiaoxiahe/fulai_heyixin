#!/usr/bin/env python3
"""
多模型集成评估系统 - 使用投票或平均置信度机制
支持加载多个训练好的DialogueGCN模型进行集成预测
支持并行预测以提高性能
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

class SingleModelPredictor:
    """单个模型预测器"""
    
    def __init__(self, model_path, device, tokenizer, bert_model):
        """初始化单个模型预测器
        
        Args:
            model_path: 模型文件路径
            device: 运行设备
            tokenizer: 共享的BERT tokenizer
            bert_model: 共享的BERT模型
        """
        self.device = device
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        
        # 情绪标签映射
        self.emotion_map = {
            0: 'NEUTRAL',
            1: 'HAPPY',
            2: 'SAD',
            3: 'ANGRY'
        }
        
        # 共享BERT模型
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        
        # 对话历史
        self.conversation_history = []
        
        # 加载模型
        self.model = None
        self.feature_dim = 768
        self.load_model(model_path)
    
    def reset_dialogue_context(self):
        """重置对话上下文"""
        self.conversation_history = []
    
    def load_model(self, model_path):
        """加载训练好的DialogueGCN模型"""
        if not MODEL_AVAILABLE:
            print(f"❌ [{self.model_name}] 无法导入DialogueGCN模型")
            return False
        
        try:
            if not os.path.exists(model_path):
                print(f"❌ [{self.model_name}] 模型文件不存在")
                return False
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                
                if 'feature_dim' in checkpoint:
                    feature_dim = checkpoint['feature_dim']
                else:
                    feature_dim = 768
                
                self.feature_dim = feature_dim
                
                config = checkpoint.get('config', {})
                D_m = feature_dim
                
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
                
                self.model = DialogueGCNModel(
                    base_model='LSTM',
                    D_m=D_m,
                    D_g=D_g,
                    D_p=D_p,
                    D_e=D_e,
                    D_h=D_h,
                    D_a=D_a,
                    graph_hidden_size=graph_h,
                    n_speakers=4,
                    max_seq_len=300,
                    window_past=10,
                    window_future=10,
                    n_classes=4,
                    listener_state=False,
                    context_attention='simple',
                    dropout_rec=0.1,
                    dropout=0.2,
                    nodal_attention=False,
                    avec=False,
                    no_cuda=(self.device == 'cpu')
                )
                
                self.model.load_state_dict(model_state)
                self.model.to(self.device)
                self.model.eval()
                return True
                
            else:
                print(f"❌ [{self.model_name}] 不支持的检查点格式")
                return False
                
        except Exception as e:
            print(f"❌ [{self.model_name}] 模型加载失败: {e}")
            return False
    
    def extract_bert_features(self, text):
        """提取BERT特征"""
        if self.bert_model is None or self.tokenizer is None:
            return torch.zeros(self.feature_dim, device=self.device)
        
        try:
            text = text.strip()
            if not text:
                return torch.zeros(self.feature_dim, device=self.device)
            
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                
                if self.feature_dim == 768:
                    features = outputs.last_hidden_state[:, 0, :].squeeze(0)
                elif self.feature_dim == 2304:
                    hidden_states = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    
                    cls_features = hidden_states[:, 0, :].squeeze(0)
                    
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    mean_features = (sum_embeddings / sum_mask).squeeze(0)
                    
                    hidden_states_clone = hidden_states.clone()
                    hidden_states_clone[input_mask_expanded == 0] = -1e9
                    max_features = torch.max(hidden_states_clone, 1)[0].squeeze(0)
                    
                    features = torch.cat([cls_features, mean_features, max_features], dim=0)
                else:
                    features = outputs.last_hidden_state[:, 0, :].squeeze(0)
                    if features.size(0) != self.feature_dim:
                        if not hasattr(self, 'feature_adapter'):
                            self.feature_adapter = torch.nn.Linear(768, self.feature_dim).to(self.device)
                        features = self.feature_adapter(features)
            
            return features
            
        except Exception as e:
            print(f"⚠️ [{self.model_name}] BERT特征提取失败: {e}")
            return torch.zeros(self.feature_dim, device=self.device)
    
    def predict_emotion(self, user_text, robot_text=""):
        """预测情绪
        
        Returns:
            Dict包含emotion, confidence, probabilities
        """
        if self.model is None:
            return None
        
        try:
            self.model.eval()
            
            # 构建对话序列
            conversation = []
            
            # 添加历史对话（最近5轮）
            for hist_text, hist_speaker in self.conversation_history[-5:]:
                conversation.append((hist_text, hist_speaker))
            
            # 添加当前对话
            if robot_text:
                conversation.append((robot_text, 1))
            conversation.append((user_text, 0))
            
            # 提取特征
            text_features = []
            speaker_features = []
            
            for text, speaker in conversation:
                text_feature = self.extract_bert_features(text)
                if text_feature.is_cuda:
                    text_feature = text_feature.cpu()
                text_features.append(text_feature.numpy())
                
                speaker_feature = [0.0] * 4
                speaker_feature[speaker] = 1.0
                speaker_features.append(speaker_feature)
            
            # 转换为张量
            text_features = torch.FloatTensor(np.array(text_features)).unsqueeze(1).to(self.device)
            speaker_features = torch.FloatTensor(np.array(speaker_features)).unsqueeze(1).to(self.device)
            
            # 创建mask
            seq_len = text_features.shape[0]
            umask = torch.ones(seq_len).unsqueeze(1).to(self.device)
            qmask = speaker_features
            
            lengths = [seq_len]
            
            # 预测
            with torch.no_grad():
                log_prob, _, _, _, _ = self.model(text_features, qmask, umask, lengths)
                
                if log_prob.dim() == 2:
                    last_logits = log_prob[-1]
                else:
                    last_logits = log_prob[-1, :]
                
                probabilities = torch.exp(last_logits)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
            
            # 构建概率字典
            probs = {}
            for i, emotion in self.emotion_map.items():
                probs[emotion] = probabilities[i].item()
            
            # 更新对话历史
            if robot_text:
                self.conversation_history.append((robot_text, 1))
            self.conversation_history.append((user_text, 0))
            
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return {
                'emotion': self.emotion_map[predicted_class],
                'confidence': confidence,
                'probabilities': probs,
                'model_name': self.model_name
            }
            
        except Exception as e:
            print(f"❌ [{self.model_name}] 预测失败: {e}")
            return None


class EnsembleModelEvaluator:
    """多模型集成评估器"""
    
    def __init__(self, model_paths: List[str], device='auto', ensemble_method='vote', parallel=True, max_workers=None):
        """初始化集成评估器
        
        Args:
            model_paths: 模型文件路径列表
            device: 设备 ('cuda', 'cpu', 或 'auto')
            ensemble_method: 集成方法 ('vote': 投票, 'avg_confidence': 平均置信度, 'weighted_avg': 加权平均)
            parallel: 是否使用并行预测 (默认True)
            max_workers: 最大并行worker数量 (默认为模型数量)
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
        
        self.ensemble_method = ensemble_method
        self.parallel = parallel
        self.max_workers = max_workers
        
        # 线程锁，用于保护BERT模型的访问
        self.bert_lock = threading.Lock()
        
        # 情绪标签
        self.emotion_labels = ['NEUTRAL', 'HAPPY', 'SAD', 'ANGRY']
        
        # 初始化BERT（所有模型共享）
        self.tokenizer = None
        self.bert_model = None
        self.load_bert()
        
        # 加载所有模型
        self.models = []
        print(f"\n正在加载 {len(model_paths)} 个模型...")
        for i, model_path in enumerate(model_paths, 1):
            print(f"\n[{i}/{len(model_paths)}] 加载模型: {model_path}")
            predictor = SingleModelPredictor(model_path, self.device, self.tokenizer, self.bert_model)
            if predictor.model is not None:
                self.models.append(predictor)
                print(f"✅ 模型加载成功")
            else:
                print(f"⚠️ 模型加载失败，跳过此模型")
        
        if len(self.models) == 0:
            print("\n❌ 没有成功加载任何模型！")
        else:
            print(f"\n✅ 成功加载 {len(self.models)}/{len(model_paths)} 个模型")
            print(f"📊 集成方法: {ensemble_method}")
            if self.parallel:
                workers = self.max_workers if self.max_workers else len(self.models)
                print(f"⚡ 并行模式: 开启 (最大{workers}个并发)")
            else:
                print(f"⚙️  并行模式: 关闭 (串行执行)")
    
    def load_bert(self):
        """加载BERT模型（共享）"""
        if not BERT_AVAILABLE:
            print("⚠️ transformers库不可用")
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
    
    def reset_dialogue_context(self):
        """重置所有模型的对话上下文"""
        for model in self.models:
            model.reset_dialogue_context()
        print("🔄 所有模型的对话上下文已重置")
    
    def ensemble_predict(self, user_text, robot_text="", verbose=False):
        """使用集成方法预测情绪
        
        Args:
            user_text: 用户输入文本
            robot_text: 机器人回复文本
            verbose: 是否显示详细信息
        
        Returns:
            Dict包含ensemble_emotion, confidence, all_predictions, method_details
        """
        if len(self.models) == 0:
            print("❌ 没有可用的模型")
            return None
        
        # 获取所有模型的预测（支持并行）
        if self.parallel:
            all_predictions = self._parallel_predict(user_text, robot_text)
        else:
            all_predictions = self._sequential_predict(user_text, robot_text)
        
        if len(all_predictions) == 0:
            print("❌ 所有模型预测失败")
            return None
        
        # 根据集成方法进行融合
        if self.ensemble_method == 'vote':
            return self._voting_ensemble(all_predictions, verbose)
        elif self.ensemble_method == 'avg_confidence':
            return self._average_confidence_ensemble(all_predictions, verbose)
        elif self.ensemble_method == 'weighted_avg':
            return self._weighted_average_ensemble(all_predictions, verbose)
        else:
            print(f"⚠️ 未知的集成方法: {self.ensemble_method}，使用投票法")
            return self._voting_ensemble(all_predictions, verbose)
    
    def _sequential_predict(self, user_text, robot_text=""):
        """串行预测（原始方法）"""
        all_predictions = []
        for model in self.models:
            result = model.predict_emotion(user_text, robot_text)
            if result:
                all_predictions.append(result)
        return all_predictions
    
    def _parallel_predict(self, user_text, robot_text=""):
        """并行预测（使用线程池）"""
        all_predictions = []
        max_workers = self.max_workers if self.max_workers else len(self.models)
        
        # 创建预测任务
        def predict_task(model):
            """单个模型的预测任务"""
            try:
                return model.predict_emotion(user_text, robot_text)
            except Exception as e:
                print(f"⚠️ 模型 {model.model_name} 预测出错: {e}")
                return None
        
        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_model = {
                executor.submit(predict_task, model): model 
                for model in self.models
            }
            
            # 收集结果
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    if result:
                        all_predictions.append(result)
                except Exception as e:
                    print(f"⚠️ 获取模型 {model.model_name} 预测结果失败: {e}")
        
        return all_predictions
    
    def _voting_ensemble(self, all_predictions, verbose=False):
        """投票集成方法"""
        # 统计每个情绪的投票数
        votes = Counter([pred['emotion'] for pred in all_predictions])
        
        # 获取票数最多的情绪
        ensemble_emotion = votes.most_common(1)[0][0]
        vote_count = votes[ensemble_emotion]
        confidence = vote_count / len(all_predictions)
        
        result = {
            'ensemble_emotion': ensemble_emotion,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'method': 'vote',
            'vote_details': dict(votes),
            'total_models': len(all_predictions)
        }
        
        if verbose:
            print(f"\n📊 投票结果:")
            for emotion, count in votes.most_common():
                print(f"   {emotion}: {count}/{len(all_predictions)} 票 ({count/len(all_predictions)*100:.1f}%)")
        
        return result
    
    def _average_confidence_ensemble(self, all_predictions, verbose=False):
        """平均置信度集成方法"""
        # 计算每个情绪的平均置信度
        avg_probs = {emotion: [] for emotion in self.emotion_labels}
        
        for pred in all_predictions:
            for emotion, prob in pred['probabilities'].items():
                avg_probs[emotion].append(prob)
        
        # 计算平均值
        final_probs = {}
        for emotion, probs in avg_probs.items():
            if probs:
                final_probs[emotion] = np.mean(probs)
            else:
                final_probs[emotion] = 0.0
        
        # 选择平均置信度最高的情绪
        ensemble_emotion = max(final_probs.items(), key=lambda x: x[1])[0]
        confidence = final_probs[ensemble_emotion]
        
        result = {
            'ensemble_emotion': ensemble_emotion,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'method': 'avg_confidence',
            'avg_probabilities': final_probs,
            'total_models': len(all_predictions)
        }
        
        if verbose:
            print(f"\n📊 平均置信度:")
            for emotion, prob in sorted(final_probs.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 30)
                print(f"   {emotion:8}: {prob:.4f} {bar}")
        
        return result
    
    def _weighted_average_ensemble(self, all_predictions, verbose=False):
        """加权平均集成方法（权重为各模型的置信度）"""
        # 计算每个情绪的加权置信度
        weighted_probs = {emotion: 0.0 for emotion in self.emotion_labels}
        total_weight = 0.0
        
        for pred in all_predictions:
            weight = pred['confidence']  # 使用模型自己的置信度作为权重
            total_weight += weight
            
            for emotion, prob in pred['probabilities'].items():
                weighted_probs[emotion] += prob * weight
        
        # 归一化
        if total_weight > 0:
            for emotion in weighted_probs:
                weighted_probs[emotion] /= total_weight
        
        # 选择加权置信度最高的情绪
        ensemble_emotion = max(weighted_probs.items(), key=lambda x: x[1])[0]
        confidence = weighted_probs[ensemble_emotion]
        
        result = {
            'ensemble_emotion': ensemble_emotion,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'method': 'weighted_avg',
            'weighted_probabilities': weighted_probs,
            'total_models': len(all_predictions),
            'total_weight': total_weight
        }
        
        if verbose:
            print(f"\n📊 加权平均置信度:")
            for emotion, prob in sorted(weighted_probs.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 30)
                print(f"   {emotion:8}: {prob:.4f} {bar}")
        
        return result
    
    def run_interactive(self):
        """交互式评估模式"""
        print("\n" + "="*80)
        print("🎯 多模型集成交互式评估")
        print("="*80)
        print(f"使用 {len(self.models)} 个训练好的DialogueGCN模型进行集成预测")
        print(f"集成方法: {self.ensemble_method}")
        print("支持多轮对话上下文")
        print("输入 'quit' 退出")
        print("输入 'clear' 清空对话历史")
        print("输入 'method <方法>' 切换集成方法 (vote/avg_confidence/weighted_avg)")
        print("="*80)
        
        total_predictions = 0
        prediction_times = []
        emotion_counts = Counter()
        
        while True:
            try:
                user_input = input("\n👤 用户: ").strip()
                
                if user_input.lower() == 'quit':
                    if total_predictions > 0:
                        print("\n" + "="*60)
                        print("📊 会话统计")
                        print("="*60)
                        print(f"总预测次数: {total_predictions}")
                        print(f"平均耗时: {np.mean(prediction_times):.2f}ms")
                        print(f"\n情绪分布:")
                        for emotion, count in emotion_counts.most_common():
                            print(f"  {emotion}: {count} ({count/total_predictions*100:.1f}%)")
                    print("\n👋 再见！")
                    break
                
                elif user_input.lower() == 'clear':
                    self.reset_dialogue_context()
                    continue
                
                elif user_input.lower().startswith('method '):
                    new_method = user_input[7:].strip()
                    if new_method in ['vote', 'avg_confidence', 'weighted_avg']:
                        self.ensemble_method = new_method
                        print(f"✅ 集成方法已切换为: {new_method}")
                    else:
                        print(f"⚠️ 无效的集成方法，请选择: vote, avg_confidence, weighted_avg")
                    continue
                
                elif not user_input:
                    print("⚠️ 请输入有效内容")
                    continue
                
                # 预测情绪
                print(f"\n🔍 正在使用 {len(self.models)} 个模型进行集成预测...")
                t0 = time.perf_counter()
                result = self.ensemble_predict(user_input, robot_text="", verbose=True)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                
                if result:
                    ensemble_emotion = result['ensemble_emotion']
                    confidence = result['confidence']
                    
                    total_predictions += 1
                    prediction_times.append(elapsed_ms)
                    emotion_counts[ensemble_emotion] += 1
                    
                    print(f"\n🎯 集成预测结果: {ensemble_emotion} (置信度: {confidence:.2%})")
                    print(f"⏱️  总耗时: {elapsed_ms:.2f}ms")
                    
                    # 显示各个模型的预测
                    print(f"\n📋 各模型预测详情:")
                    for i, pred in enumerate(result['all_predictions'], 1):
                        model_name = pred['model_name']
                        emotion = pred['emotion']
                        conf = pred['confidence']
                        match = "✅" if emotion == ensemble_emotion else "❌"
                        print(f"   {match} 模型{i} ({model_name}): {emotion} (置信度: {conf:.2%})")
                else:
                    print("❌ 预测失败")
                    continue
                
                # 可选的机器人回复
                robot_input = input("\n🤖 机器人回复 (可选，直接回车跳过): ").strip()
                if robot_input:
                    for model in self.models:
                        model.conversation_history.append((robot_input, 1))
                    print(f"✅ 机器人回复已记录到所有模型")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
                import traceback
                traceback.print_exc()
    
    def evaluate_on_file(self, data_path: str = 'improved_test_data.xlsx'):
        """使用测试文件进行批量评估"""
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
                print("❌ 不支持的文件格式")
                return
        except Exception as e:
            print(f"❌ 读取测试数据失败: {e}")
            return
        
        required_cols = {'dialogue_id', 'speaker', 'utterance', 'emotion_label'}
        if not required_cols.issubset(set(df.columns)):
            print(f"❌ 测试数据缺少必要列")
            return
        
        df = df.copy()
        df['speaker'] = df['speaker'].astype(str)
        
        user_rows = df[(df['speaker'] == 'user') & df['emotion_label'].notna() & (df['emotion_label'].astype(str) != '')]
        robot_rows = df[(df['speaker'] == 'robot') & df['emotion_label'].notna() & (df['emotion_label'].astype(str) != '')]
        print(f"\n🧪 多模型集成评估...")
        print(f"总记录数: {len(df)} | 用户(有标签): {len(user_rows)} | 机器人(有标签): {len(robot_rows)}")
        print(f"使用模型数: {len(self.models)} | 集成方法: {self.ensemble_method}")
        
        label_map = {
            'neutral': 'NEUTRAL',
            'happy': 'HAPPY',
            'sad': 'SAD',
            'angry': 'ANGRY'
        }
        valid_labels = ['NEUTRAL', 'HAPPY', 'SAD', 'ANGRY']
        
        y_true, y_pred, times = [], [], []
        
        # 按对话分组
        for dialogue_id, group in df.groupby('dialogue_id', sort=False):
            print(f"\n=== 对话 {dialogue_id} 开始，{len(group)} 条 ===")
            self.reset_dialogue_context()
            
            group = group.reset_index(drop=True)
            for idx, row in group.iterrows():
                speaker = str(row['speaker']).strip().lower()
                utterance = str(row['utterance']) if not pd.isna(row['utterance']) else ''
                true_label_raw = row.get('emotion_label')
                true_label = '' if pd.isna(true_label_raw) else str(true_label_raw)
                
                if speaker == 'robot':
                    if utterance.strip():
                        for model in self.models:
                            model.conversation_history.append((utterance.strip(), 1))
                    
                    if true_label.strip():
                        t0 = time.perf_counter()
                        result = self.ensemble_predict(utterance, robot_text="", verbose=False)
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        
                        if result:
                            pred = result['ensemble_emotion']
                            confidence = result['confidence']
                        else:
                            pred = 'NEUTRAL'
                            confidence = 0.0
                        
                        mapped_true = label_map.get(true_label.lower(), true_label.upper())
                        y_true.append(mapped_true)
                        y_pred.append(pred)
                        times.append(elapsed_ms)
                        
                        match_symbol = "✅" if pred == mapped_true else "❌"
                        print(f"{match_symbol} Robot: {utterance[:50]}...")
                        print(f"   耗时: {elapsed_ms:.1f}ms | 预测: {pred} (置信度: {confidence:.2%}) | 实际: {mapped_true}")
                    continue
                
                if speaker == 'user':
                    if not true_label.strip():
                        if utterance.strip():
                            for model in self.models:
                                model.conversation_history.append((utterance.strip(), 0))
                        continue
                    
                    t0 = time.perf_counter()
                    result = self.ensemble_predict(utterance, robot_text="", verbose=False)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    
                    if result:
                        pred = result['ensemble_emotion']
                        confidence = result['confidence']
                    else:
                        pred = 'NEUTRAL'
                        confidence = 0.0
                    
                    mapped_true = label_map.get(true_label.lower(), true_label.upper())
                    y_true.append(mapped_true)
                    y_pred.append(pred)
                    times.append(elapsed_ms)
                    
                    match_symbol = "✅" if pred == mapped_true else "❌"
                    print(f"{match_symbol} User: {utterance[:50]}...")
                    print(f"   耗时: {elapsed_ms:.1f}ms | 预测: {pred} (置信度: {confidence:.2%}) | 实际: {mapped_true}")
        
        # 汇总指标
        if not y_true:
            print("\n❌ 没有可评估的标注数据")
            return
        
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        accuracy = float((y_true_np == y_pred_np).mean())
        
        print("\n" + "="*80)
        print("📊 多模型集成评估结果汇总")
        print("="*80)
        print(f"集成方法: {self.ensemble_method}")
        print(f"使用模型数: {len(self.models)}")
        print(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"总样本数: {len(y_true)}")
        
        # 各类情绪的详细统计
        print("\n" + "="*80)
        print("📈 各类情绪详细统计")
        print("="*80)
        
        if _SK_AVAILABLE:
            try:
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true_np, y_pred_np, labels=valid_labels, zero_division=0
                )
                
                print(f"{'情绪类别':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'支持数':<8}")
                print("-" * 60)
                
                for i, lbl in enumerate(valid_labels):
                    print(f"{lbl:<12} {precision[i]:<8.4f} {recall[i]:<8.4f} {f1[i]:<8.4f} {int(support[i]):<8}")
                
                macro_precision = np.mean(precision)
                macro_recall = np.mean(recall)
                macro_f1 = np.mean(f1)
                
                print("-" * 60)
                print(f"{'宏平均':<12} {macro_precision:<8.4f} {macro_recall:<8.4f} {macro_f1:<8.4f} {'-':<8}")
                
            except Exception as e:
                print(f"  计算详细指标失败: {e}")
        
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
            except Exception as e:
                print(f"\n混淆矩阵计算失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多模型集成评估')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                       help='模型文件路径列表（空格分隔）')
    parser.add_argument('--mode', type=str, choices=['interactive', 'file'],
                       default='interactive',
                       help='评估模式: interactive(交互式) 或 file(文件批量评估)')
    parser.add_argument('--test_file', type=str,
                       default='improved_test_data.xlsx',
                       help='测试数据文件路径（mode=file时使用）')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                       default='auto',
                       help='运行设备')
    parser.add_argument('--ensemble_method', type=str, 
                       choices=['vote', 'avg_confidence', 'weighted_avg'],
                       default='vote',
                       help='集成方法: vote(投票), avg_confidence(平均置信度), weighted_avg(加权平均)')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='使用并行预测（默认开启）')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false',
                       help='禁用并行预测，使用串行模式')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='最大并行worker数量（默认为模型数量）')
    
    args = parser.parse_args()
    
    # 检查模型文件
    valid_model_paths = []
    for model_path in args.model_paths:
        if os.path.exists(model_path):
            valid_model_paths.append(model_path)
            print(f"✅ 找到模型: {model_path}")
        else:
            print(f"⚠️ 模型文件不存在，跳过: {model_path}")
    
    if len(valid_model_paths) == 0:
        print("❌ 没有找到任何有效的模型文件")
        return
    
    print(f"\n将使用 {len(valid_model_paths)} 个模型进行集成评估")
    
    # 创建集成评估器
    evaluator = EnsembleModelEvaluator(
        valid_model_paths, 
        device=args.device,
        ensemble_method=args.ensemble_method,
        parallel=args.parallel,
        max_workers=args.max_workers
    )
    
    if len(evaluator.models) == 0:
        print("❌ 没有成功加载任何模型，无法继续")
        return
    
    # 根据模式运行评估
    if args.mode == 'interactive':
        evaluator.run_interactive()
    elif args.mode == 'file':
        evaluator.evaluate_on_file(args.test_file)

if __name__ == "__main__":
    main()

