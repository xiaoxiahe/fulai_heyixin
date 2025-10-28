#!/usr/bin/env python3
"""
最终混合情绪识别系统
结合DialogueGCN和大模型，实现智能情绪判断
"""
import sys
sys.path.append('DialogueGCN')
sys.path.append('code')

import torch
import torch.nn as nn
import numpy as np
import os
import re
import jieba
import time
import json
import uuid
from collections import Counter, deque
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# 导入数据库相关
try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    print("⚠️ PyMySQL库未安装，将无法存储到数据库")

# 导入BERT相关库
try:
    from transformers import AutoTokenizer, AutoModel
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️ BERT库未安装，将使用简化特征提取")

# 导入大模型相关
try:
    from volcenginesdkarkruntime import Ark
    ARK_AVAILABLE = True
except ImportError:
    ARK_AVAILABLE = False
    print("⚠️ 大模型库未安装，将使用简化大模型模拟")

class SimpleDialogueGCN(nn.Module):
    """简化的DialogueGCN模型"""
    
    def __init__(self, input_dim=768, hidden_dim=200, num_classes=5, dropout=0.5):
        super(SimpleDialogueGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 文本编码器 (LSTM)
        self.text_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        
        # 说话者编码器
        self.speaker_encoder = nn.Linear(2, hidden_dim)
        
        # 说话者投影层
        self.speaker_projection = nn.Linear(hidden_dim, hidden_dim * 2)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, text_features, visual_features, audio_features, speaker_features, mask):
        """前向传播"""
        # 检查text_features的维度
        if text_features.dim() == 2:
            batch_size = text_features.shape[0]
            seq_len = 1
            text_features = text_features.unsqueeze(1)
        elif text_features.dim() == 3:
            batch_size, seq_len, _ = text_features.shape
        else:
            raise ValueError(f"不支持的text_features维度: {text_features.dim()}")
        
        # 文本编码
        text_output, _ = self.text_encoder(text_features)
        
        # 说话者编码
        speaker_output = self.speaker_encoder(speaker_features)
        speaker_output = self.speaker_projection(speaker_output)
        
        # 特征融合
        combined_features = text_output + speaker_output
        
        # 注意力机制
        attended_features, _ = self.attention(
            combined_features, combined_features, combined_features,
            key_padding_mask=~mask.bool() if mask is not None else None
        )
        
        # 分类
        logits = self.classifier(attended_features)
        
        # 应用掩码
        if mask is not None:
            min_len = min(mask.shape[1], logits.shape[1])
            mask = mask[:, :min_len]
            logits = logits[:, :min_len, :]
            mask_expanded = mask.unsqueeze(-1).expand_as(logits)
            logits = logits * mask_expanded
        
        return logits

class LargeModelClient:
    """大模型客户端"""
    
    def __init__(self):
        self.client = None
        if ARK_AVAILABLE:
            try:
                os.environ["ARK_API_KEY"] = "be62df6f-1828-47a4-84f1-2932c111bc64"
                # 优化超时设置，减少到30秒
                self.client = Ark(api_key=os.environ.get("ARK_API_KEY"), timeout=30)
            except Exception as e:
                print(f"⚠️ 大模型初始化失败: {e}")
                self.client = None
    
    def analyze_emotion(self, text: str, context: str = "") -> Dict[str, Any]:
        """使用大模型分析情绪（优化版）"""
        if not self.client:
            return self._simulate_large_model(text)
        
        try:
            # 完全参考emotion_based_text.py的简化消息结构
            messages = [
                {
                    "role": "system",
                    "content": """你是一个严格的文本情感分类器。必须只输出一个严格的 JSON 对象，不要输出多余文本、解释或反引号。
                    ## 你的任务
                    根据用户输入的文本，判断其表达的情感。
                    ## 输出要求
                    1. 输出必须为严格 JSON，且仅包含以下字段：
                    {"emotion": string, "reason": string}
                    2. `emotion` 从 ["NEUTRAL", "SAD", "ANGRY", "HAPPY"， "SURPRISE"] 中选择。
                    3. `reason` 字段提供简要判断依据（20字内）。
                    4. 若难以判断，统一判为"NEUTRAL"。
                    """
                },
                {
                    "role": "user",
                    "content": f"待分类文本：「{text}」"
                }
            ]
            
            # 如果有上下文，简化添加方式
            if context:
                messages[1]["content"] = f"上下文：{context}\n待分类文本：「{text}」"
            
            # 添加性能计时
            t0 = time.perf_counter()
            
            response = self.client.chat.completions.create(
                model="doubao-seed-1.6-250615",
                messages=messages,
                thinking={"type": "disabled"},
                response_format={"type": "json_object"}
            )
            
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            print(f"⏱️ 大模型响应时间: {elapsed_ms}ms")
            
            # 解析响应，完全参考emotion_based_text.py的解析方式
            msg = response.choices[0].message
            try:
                result = getattr(msg, "parsed", None) or json.loads(getattr(msg, "content", msg))
            except Exception as e:
                print(f"JSON 解析失败: {e}")
                result = {"emotion": "NEUTRAL", "reason": "解析失败"}
            
            # 确保情绪标签正确
            emotion = str(result.get("emotion", "NEUTRAL")).upper()
            if emotion not in ["NEUTRAL", "SAD", "ANGRY", "HAPPY"]:
                emotion = "NEUTRAL"
            
            # 生成概率分布（基于情绪标签）
            probs_filled = {k: 0.1 for k in ["NEUTRAL", "SAD", "ANGRY", "HAPPY"]}
            probs_filled[emotion] = 0.7  # 主要情绪占70%
            
            # 计算置信度
            confidence = 0.7 if emotion != "NEUTRAL" else 0.5
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "emotion_probs": probs_filled,
                "reason": str(result.get("reason", ""))[:30],
                "source": "large_model",
                "response_time_ms": elapsed_ms
            }
            
        except Exception as e:
            print(f"⚠️ 大模型调用失败: {e}")
            print(f"   回退到模拟大模型")
            return self._simulate_large_model(text)
    
class FinalEmotionSystem:
    """最终混合情绪识别系统"""
    
    def __init__(self, model_path, device='auto', session_id=None):
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
        
        # 情绪标签映射
        self.emotion_map = {
            0: 'NEUTRAL',    # neutral
            1: 'HAPPY',      # happy
            2: 'SAD',        # sad
            3: 'ANGRY',      # angry
            4: 'SURPRISE'    # surprise
        }
        
        self.reverse_emotion_map = {v: k for k, v in self.emotion_map.items()}
        
        # 对话历史
        self.conversation_history = []
        self.turn_count = 0
        self.previous_gcn_probs = None
        
        # 会话ID
        self.session_id = session_id or str(uuid.uuid4())
        
        # 数据库相关
        self.db_available = MYSQL_AVAILABLE
        if self.db_available:
            self.init_database()
        
        # 初始化组件
        self.model = None
        self.tokenizer = None
        self.bert_model = None
        self.large_model_client = LargeModelClient()
        
        # 加载模型和BERT
        self.load_model(model_path)
        self.load_bert()
    
    def reset_dialogue_context(self):
        """重置对话上下文历史窗口"""
        self.conversation_history = []
        self.turn_count = 0
        self.previous_gcn_probs = None
        # 重置当前对话ID
        if hasattr(self, 'current_dialogue_id'):
            delattr(self, 'current_dialogue_id')
        print("🔄 对话上下文已重置")
    
    def init_database(self):
        """初始化数据库"""
        try:
            conn = self._mysql_conn()
            with conn.cursor() as c:
                # 创建turn_emotions表
                c.execute("""
                    CREATE TABLE IF NOT EXISTS turn_emotions (
                        session_id VARCHAR(128) NOT NULL,
                        turn_id VARCHAR(128) NOT NULL,
                        role VARCHAR(32) NOT NULL,
                        text TEXT,
                        emotion VARCHAR(16) NOT NULL,
                        probs_json JSON,
                        confidence DOUBLE,
                        source VARCHAR(64),
                        reason TEXT,
                        created_at DOUBLE,
                        INDEX idx_turn_session(session_id),
                        INDEX idx_turn_time(created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)
                
                # 检查并添加缺失的字段
                try:
                    c.execute("ALTER TABLE turn_emotions ADD COLUMN source VARCHAR(64) AFTER confidence")
                    print("✅ 添加source字段")
                except Exception:
                    pass  # 字段已存在
                
                try:
                    c.execute("ALTER TABLE turn_emotions ADD COLUMN reason TEXT AFTER source")
                    print("✅ 添加reason字段")
                except Exception:
                    pass  # 字段已存在
                
                # 创建session_state表
                c.execute("""
                    CREATE TABLE IF NOT EXISTS session_state (
                        session_id VARCHAR(128) NOT NULL PRIMARY KEY,
                        stable_emotion VARCHAR(16) NOT NULL,
                        stable_confidence DOUBLE,
                        last_updated_at DOUBLE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)
                
                # 创建shift_events表
                c.execute("""
                    CREATE TABLE IF NOT EXISTS shift_events (
                        session_id VARCHAR(128) NOT NULL,
                        from_emotion VARCHAR(16) NOT NULL,
                        to_emotion VARCHAR(16) NOT NULL,
                        at_turn_id VARCHAR(128) NOT NULL,
                        evidence JSON,
                        delta_conf DOUBLE,
                        created_at DOUBLE,
                        INDEX idx_shift_session(session_id),
                        INDEX idx_shift_time(created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)
            
            conn.close()
            print("✅ 数据库初始化成功")
            
        except Exception as e:
            print(f"❌ 数据库初始化失败: {e}")
            self.db_available = False
    
    def _mysql_conn(self):
        """获取MySQL连接"""
        if not MYSQL_AVAILABLE:
            raise RuntimeError("未安装 PyMySQL，请先安装: pip install pymysql")
        
        host = os.environ.get("MYSQL_HOST", "127.0.0.1")
        port = int(os.environ.get("MYSQL_PORT", "3306"))
        user = os.environ.get("MYSQL_USER", "root")
        password = os.environ.get("MYSQL_PASSWORD", "HYXjy9920")
        database = os.environ.get("MYSQL_DB", "logs")
        charset = "utf8mb4"
        
        return pymysql.connect(
            host=host, port=port, user=user, password=password, 
            database=database, charset=charset, autocommit=True
        )
    
    def save_emotion_result(self, turn_id, role, text, emotion, confidence, probabilities, source, reason):
        """保存情绪识别结果到数据库"""
        if not self.db_available:
            return
        
        try:
            conn = self._mysql_conn()
            with conn.cursor() as c:
                c.execute("""
                    INSERT INTO turn_emotions(
                        session_id, turn_id, role, text, emotion, probs_json, 
                        confidence, source, reason, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.session_id,
                    turn_id,
                    role,
                    text,
                    emotion,
                    json.dumps(probabilities, ensure_ascii=False),
                    confidence,
                    source,
                    reason,
                    time.time()
                ))
            conn.close()
            
        except Exception as e:
            print(f"⚠️ 保存情绪结果到数据库失败: {e}")
    
    def load_model(self, model_path):
        """加载训练好的DialogueGCN模型"""
        print("正在加载DialogueGCN模型...")
        
        try:
            # 创建模型实例 - 使用与预训练模型一致的hidden_dim=100
            self.model = SimpleDialogueGCN(
                input_dim=768,
                hidden_dim=200,  # 修改为100以匹配预训练模型
                num_classes=5,
                dropout=0.5
            )
            # 先将模型迁移到目标设备，确保后续权重/缓冲区设备一致
            self.model.to(self.device)

            # 加载模型权重
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state_dict)
                print(f"✅ 成功加载模型: {model_path}")
            else:
                print(f"❌ 模型文件不存在: {model_path}")
                return False
            
            self.model.to(self.device)
            self.model.eval()
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def load_bert(self):
        """加载chinese-roberta-wwm-ext模型"""
        if not BERT_AVAILABLE:
            print("⚠️ transformers库不可用，使用简化特征提取")
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
            print("✅ chinese-roberta-wwm-ext模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ BERT模型加载失败: {e}")
            return False
    
    def extract_bert_features(self, text):
        """提取BERT特征"""
        if self.bert_model is None or self.tokenizer is None:
            return self.extract_simple_features(text)
        
        try:
            text = text.strip()
            if not text:
                return torch.zeros(768, device=self.device)
            
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
                features = outputs.last_hidden_state[:, 0, :]
            
            return features.squeeze(0)
            
        except Exception as e:
            print(f"⚠️ BERT特征提取失败: {e}")
            return self.extract_simple_features(text)
    
    # def extract_simple_features(self, text):
    #     """简化特征提取"""
    #     text_lower = text.lower()
        
    #     emotion_keywords = {
    #         'happy': ['开心', '高兴', '快乐', '兴奋', '满意', '喜欢', '爱', '好', '棒', '赞', '哈哈', '嘿嘿', '耶', '太棒了', '完美'],
    #         'sad': ['难过', '伤心', '痛苦', '失望', '沮丧', '哭', '泪', '坏', '差', '糟糕', '唉', '唉声叹气', '郁闷', '烦躁'],
    #         'angry': ['生气', '愤怒', '恼火', '讨厌', '恨', '烦', '气', '怒', '火', '暴躁', '可恶', '该死', '混蛋', '气死我了'],
    #         'surprise': ['惊讶', '震惊', '意外', '没想到', '竟然', '居然', '天哪', '哇', '哦', '啊', '什么', '真的吗', '不会吧']
    #     }
        
    #     emotion_scores = {emotion: 0.0 for emotion in emotion_keywords.keys()}
    #     for emotion, keywords in emotion_keywords.items():
    #         for keyword in keywords:
    #             if keyword in text_lower:
    #                 emotion_scores[emotion] += 1.0
        
    #     if max(emotion_scores.values()) > 0:
    #         main_emotion = max(emotion_scores, key=emotion_scores.get)
    #     else:
    #         main_emotion = 'neutral'
        
    #     # 生成特征向量
    #     feature = np.zeros(768)
        
    #     emotion_centers = {
    #         'happy': 0.8,
    #         'sad': -0.8,
    #         'angry': -0.6,
    #         'surprise': 0.4,
    #         'neutral': 0.0
    #     }
        
    #     center = emotion_centers[main_emotion]
    #     intensity = min(max(emotion_scores.values()) / 5.0, 1.0)
        
    #     # 前100维：基于主要情绪
    #     for i in range(100):
    #         feature[i] = center * intensity + np.random.normal(0, 0.1)
        
    #     # 100-200维：基于情绪分布
    #     for i, (emotion, score) in enumerate(emotion_scores.items()):
    #         start_idx = 100 + i * 25
    #         end_idx = start_idx + 25
    #         for j in range(start_idx, min(end_idx, 200)):
    #             feature[j] = score / 5.0 + np.random.normal(0, 0.05)
        
    #     # 200-300维：基于文本特征
    #     text_length = len(text)
    #     word_count = len(jieba.lcut(text))
        
    #     for i in range(200, 300):
    #         if i < 250:
    #             feature[i] = min(text_length / 100.0, 1.0) + np.random.normal(0, 0.05)
    #         else:
    #             feature[i] = min(word_count / 50.0, 1.0) + np.random.normal(0, 0.05)
        
    #     # 300-768维：随机特征
    #     for i in range(300, 768):
    #         feature[i] = np.random.normal(0, 0.2)
        
    #     return torch.FloatTensor(feature).to(self.device)
    
    def get_dialoguegcn_prediction(self, user_text, robot_text=""):
        """获取DialogueGCN预测结果"""
        if self.model is None:
            return None
        
        try:
            # 双重保险：确保模型已在目标设备上
            self.model.to(self.device)
            self.model.eval()
            # 构建对话序列
            conversation = []
            
            # 添加历史对话
            for hist_text, hist_speaker in self.conversation_history[-5:]:
                conversation.append((hist_text, hist_speaker))
            
            # 如果是第一句话（没有历史），添加一个中性的虚拟上下文
            # 这可以减少模型对第一句话的SAD偏向
            if len(conversation) == 0 and not robot_text:
                # 添加一个中性的机器人问候作为虚拟上下文
                virtual_context = "您好，我是您的智能助手，很高兴为您服务。"
                conversation.append((virtual_context, 1))  # robot
            
            # 添加当前对话
            if robot_text:
                conversation.append((robot_text, 1))  # robot
            conversation.append((user_text, 0))  # user
            
            # 提取特征
            text_features = []
            speaker_features = []
            
            for text, speaker in conversation:
                text_feature = self.extract_bert_features(text)
                if text_feature.is_cuda:
                    text_feature = text_feature.cpu()
                text_features.append(text_feature.numpy())
                
                speaker_feature = [0.0, 0.0]
                speaker_feature[speaker] = 1.0
                speaker_features.append(speaker_feature)
            
            # 转换为张量
            text_features = torch.FloatTensor(np.array(text_features)).unsqueeze(0).to(self.device)
            speaker_features = torch.FloatTensor(np.array(speaker_features)).unsqueeze(0).to(self.device)
            
            # 创建虚拟特征
            seq_len = text_features.shape[1]
            visual_features = torch.zeros(1, seq_len, 100).to(self.device)
            audio_features = torch.zeros(1, seq_len, 100).to(self.device)
            mask = torch.ones(1, seq_len).to(self.device)
            
            # 预测
            with torch.no_grad():
                logits = self.model(text_features, visual_features, audio_features, speaker_features, mask)
                last_logits = logits[0, -1, :]
                probabilities = torch.softmax(last_logits, dim=0)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
            
            # 构建概率字典
            probs = {}
            for i, emotion in self.emotion_map.items():
                probs[emotion] = probabilities[i].item()
            
            return {
                'emotion': self.emotion_map[predicted_class],
                'confidence': confidence,
                'probabilities': probs,
                'source': 'dialoguegcn'
            }
            
        except Exception as e:
            print(f"❌ DialogueGCN预测失败: {e}")
            return None
    
    def update_gcn_confidence(self, gcn_result, large_model_result):
        """适度更新DialogueGCN的置信度"""
        if not gcn_result or not large_model_result:
            return gcn_result
        
        # 获取大模型结果
        lm_emotion = large_model_result['emotion']
        lm_confidence = large_model_result['confidence']
        
        # 适度调整GCN结果，不要过度修改
        updated_probs = gcn_result['probabilities'].copy()
        
        # 如果大模型结果与GCN不同，适度调整
        if lm_emotion != gcn_result['emotion']:
            # 增加大模型预测的情绪概率
            adjustment = min(0.15, lm_confidence * 0.2)  # 最多调整15%
            updated_probs[lm_emotion] = min(1.0, updated_probs[lm_emotion] + adjustment)
            
            # 相应减少其他情绪的概率
            total_other = sum(updated_probs[emotion] for emotion in updated_probs if emotion != lm_emotion)
            if total_other > 0:
                for emotion in updated_probs:
                    if emotion != lm_emotion:
                        updated_probs[emotion] *= (1 - adjustment / total_other)
        
        # 重新计算置信度和预测情绪
        max_emotion = max(updated_probs, key=updated_probs.get)
        max_confidence = updated_probs[max_emotion]
        
        return {
            'emotion': max_emotion,
            'confidence': max_confidence,
            'probabilities': updated_probs,
            'source': 'dialoguegcn_updated'
        }
    
    def predict_emotion(self, user_text, robot_text="", dialogue_id=None):
        """混合情绪预测主函数 - 在用户输入后立即判断情绪
        
        Args:
            user_text: 用户输入文本
            robot_text: 机器人回复文本（可选）
            dialogue_id: 对话ID，如果提供且与当前不同，将重置上下文
        """
        # 检查是否需要重置对话上下文
        if dialogue_id is not None and hasattr(self, 'current_dialogue_id'):
            if self.current_dialogue_id != dialogue_id:
                print(f"🔄 检测到新对话ID: {dialogue_id}，重置上下文历史窗口")
                self.reset_dialogue_context()
                self.current_dialogue_id = dialogue_id
        elif dialogue_id is not None:
            self.current_dialogue_id = dialogue_id
        
        self.turn_count += 1
        turn_id = str(int(time.time() * 1000))
        
        # 1. 第一句话都默认用大模型来判断
        if self.turn_count == 1:
            print("🔄 第一句话，使用大模型判断...")
            context = " | ".join([f"{'机器人' if speaker == 1 else '用户'}: {text}" for text, speaker in self.conversation_history[-8:]])
            if robot_text:
                context += f" | 机器人: {robot_text}"
            
            print(f"📤 发送给大模型的内容:")
            print(f"   当前用户输入: {user_text}")
            print(f"   上下文历史: {context if context else '无'}")
            
            result = self.large_model_client.analyze_emotion(user_text, context)
            
            # 同时获取DialogueGCN结果用于后续比较
            gcn_result = self.get_dialoguegcn_prediction(user_text, robot_text)
            if gcn_result:
                # 保存DialogueGCN概率用于下次比较
                self.previous_gcn_probs = gcn_result['probabilities'].copy()
                print(f"🔍 同时获取DialogueGCN结果用于后续比较: {gcn_result['emotion']} (置信度: {gcn_result['confidence']:.2%})")
            
            # 更新对话历史
            if robot_text:
                self.conversation_history.append((robot_text, 1))
            self.conversation_history.append((user_text, 0))
            
            # 保存到数据库
            self.save_emotion_result(
                turn_id, "user", user_text, result['emotion'], 
                result['confidence'], result['emotion_probs'], 
                'large_model_first_turn', result['reason']
            )
            
            return {
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'probabilities': result['emotion_probs'],
                'source': 'large_model_first_turn',
                'reason': result['reason']
            }
        
        # 获取DialogueGCN预测结果
        gcn_result = self.get_dialoguegcn_prediction(user_text, robot_text)
        if not gcn_result:
            print("❌ DialogueGCN预测失败，使用大模型")
            context = " | ".join([f"{'机器人' if speaker == 1 else '用户'}: {text}" for text, speaker in self.conversation_history[-8:]])
            
            print(f"📤 发送给大模型的内容:")
            print(f"   当前用户输入: {user_text}")
            print(f"   上下文历史: {context if context else '无'}")
            
            result = self.large_model_client.analyze_emotion(user_text, context)
            
            # 保存到数据库
            self.save_emotion_result(
                turn_id, "user", user_text, result['emotion'], 
                result['confidence'], result['emotion_probs'], 
                'large_model_fallback', result['reason']
            )
            
            return {
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'probabilities': result['emotion_probs'],
                'source': 'large_model_fallback',
                'reason': result['reason']
            }
        
        gcn_emotion = gcn_result['emotion']
        gcn_probs = gcn_result['probabilities']
        neutral_prob = gcn_probs.get('NEUTRAL', 0.0)
        
        # 2. DialogueGCN判定结果不为neutral时才为dialoguegcn结果
        if gcn_emotion != 'NEUTRAL':
            # 检查是否有其他情绪上升3%
            other_emotion_increased = False
            if self.previous_gcn_probs:
                print(f"🔍 检查其他情绪是否上升3%:")
                for emotion in gcn_probs:
                    if emotion != gcn_emotion:  # 检查除当前预测情绪外的其他情绪
                        current_prob = gcn_probs.get(emotion, 0.0)
                        previous_prob = self.previous_gcn_probs.get(emotion, 0.0)
                        change = current_prob - previous_prob
                        print(f"   {emotion}: {previous_prob:.3f} -> {current_prob:.3f} (变化: {change:+.3f})")
                        if change > 0.03:  # 3%阈值
                            other_emotion_increased = True
                            print(f"   ✅ {emotion} 上升超过3%: {change:.3f}")
                            break
                if not other_emotion_increased:
                    print(f"   ❌ 没有其他情绪上升超过3%")
            else:
                print(f"   ⚠️ 没有之前的历史概率数据，直接使用DialogueGCN结果")
            
            # 只有当没有其他情绪上升3%时才使用DialogueGCN结果
            if not other_emotion_increased:
                print(f"🎯 DialogueGCN非neutral结果且无其他情绪上升: {gcn_emotion}")
                # 更新对话历史
                if robot_text:
                    self.conversation_history.append((robot_text, 1))
                self.conversation_history.append((user_text, 0))
                
                # 保存当前概率用于下次比较
                self.previous_gcn_probs = gcn_probs.copy()
                
                return {
                    'emotion': gcn_emotion,
                    'confidence': gcn_result['confidence'],
                    'probabilities': gcn_probs,
                    'source': 'dialoguegcn_non_neutral_stable',
                    'reason': f"DialogueGCN预测为{gcn_emotion}且无其他情绪上升"
                }
            else:
                print(f"🔄 检测到其他情绪上升，调用大模型重新判断...")
                context = " | ".join([f"{'机器人' if speaker == 1 else '用户'}: {text}" for text, speaker in self.conversation_history[-8:]])
                
                print(f"📤 发送给大模型的内容:")
                print(f"   当前用户输入: {user_text}")
                print(f"   上下文历史: {context if context else '无'}")
                
                lm_result = self.large_model_client.analyze_emotion(user_text, context)
                
                print(f"🔍 大模型重新判断结果: {lm_result['emotion']} (置信度: {lm_result['confidence']:.2%})")
                print(f" 🔍 大模型判断原因：{lm_result['reason']}")
                # 更新对话历史
                if robot_text:
                    self.conversation_history.append((robot_text, 1))
                self.conversation_history.append((user_text, 0))
                
                # 保存当前概率用于下次比较
                self.previous_gcn_probs = gcn_probs.copy()
                
                return {
                    'emotion': lm_result['emotion'],
                    'confidence': lm_result['confidence'],
                    'probabilities': lm_result['emotion_probs'],
                    'source': 'large_model_other_emotion_rising',
                    'reason': f"检测到其他情绪上升，大模型重新判断为{lm_result['emotion']}"
                }
        
        # 3. 结果为neutral，但是|neutral-other|≤5时，返回第二高的
        other_probs = {k: v for k, v in gcn_probs.items() if k != 'NEUTRAL'}
        if other_probs:
            max_other_prob = max(other_probs.values())
            max_other_emotion = max(other_probs, key=other_probs.get)
            
            if abs(neutral_prob - max_other_prob) <= 0.05:  # 5%阈值
                print(f"🔄 neutral与其他情绪接近，返回第二高: {max_other_emotion}")
                # 更新对话历史
                if robot_text:
                    self.conversation_history.append((robot_text, 1))
                self.conversation_history.append((user_text, 0))
                
                # 保存当前概率用于下次比较
                self.previous_gcn_probs = gcn_probs.copy()
                
                return {
                    'emotion': max_other_emotion,
                    'confidence': max_other_prob,
                    'probabilities': gcn_probs,
                    'source': 'dialoguegcn_second_highest',
                    'reason': f"neutral与{max_other_emotion}接近，选择{max_other_emotion}"
                }
        
        # 4. |neutral-other|>5时，如果某个置信度增加超过2，调用大模型，否则还是输出gcn判断结果
        if abs(neutral_prob - max_other_prob) > 0.05:
            # 检查是否有置信度显著增加
            confidence_increased = False
            if self.previous_gcn_probs:
                print(f"🔍 检查置信度变化:")
                print(f"   当前概率: {gcn_probs}")
                print(f"   之前概率: {self.previous_gcn_probs}")
                for emotion in other_probs:
                    current_prob = gcn_probs.get(emotion, 0.0)
                    previous_prob = self.previous_gcn_probs.get(emotion, 0.0)
                    change = current_prob - previous_prob
                    print(f"   {emotion}: {previous_prob:.3f} -> {current_prob:.3f} (变化: {change:+.3f})")
                    if change > 0.02:  # 2%阈值
                        confidence_increased = True
                        print(f"   ✅ {emotion} 置信度显著增加 {change:.3f}")
                        break
                if not confidence_increased:
                    print(f"   ❌ 没有检测到置信度显著增加")
            else:
                print(f"   ⚠️ 没有之前的历史概率数据")
            
            if confidence_increased:
                print("🔄 检测到置信度显著增加，调用大模型重新判断当前轮...")
                context = " | ".join([f"{'机器人' if speaker == 1 else '用户'}: {text}" for text, speaker in self.conversation_history[-8:]])
                
                print(f"📤 发送给大模型的内容:")
                print(f"   当前用户输入: {user_text}")
                print(f"   上下文历史: {context if context else '无'}")
                
                lm_result = self.large_model_client.analyze_emotion(user_text, context)
                
                print(f"🔍 大模型重新判断结果: {lm_result['emotion']} (置信度: {lm_result['confidence']:.2%})")
                print(f" 🔍 大模型判断原因：{lm_result['reason']}")
                
                # 5. 大模型判断结果在other上升list内，直接使用大模型结果，否则使用gcn和大模型综合下来的结果
                lm_emotion = lm_result['emotion']
                if lm_emotion in other_probs:
                    print(f"🎯 大模型结果在上升列表中: {lm_emotion}")
                    # 6. 调用完大模型需要适当更新dialogue的置信度
                    updated_gcn = self.update_gcn_confidence(gcn_result, lm_result)
                    
                    # 更新对话历史
                    if robot_text:
                        self.conversation_history.append((robot_text, 1))
                    self.conversation_history.append((user_text, 0))
                    
                    # 保存当前概率用于下次比较
                    self.previous_gcn_probs = updated_gcn['probabilities'].copy()
                    
                    return {
                        'emotion': lm_emotion,
                        'confidence': lm_result['confidence'],
                        'probabilities': updated_gcn['probabilities'],
                        'source': 'large_model_in_rising_list',
                        'reason': f"大模型重新判断{lm_emotion}，在上升列表中"
                    }
                else:
                    print(f"🔄 大模型结果不在上升列表中，综合GCN和大模型结果")
                    # 综合结果
                    combined_probs = {}
                    for emotion in gcn_probs:
                        gcn_prob = gcn_probs[emotion]
                        lm_prob = lm_result['emotion_probs'].get(emotion, 0.0)
                        combined_probs[emotion] = (gcn_prob + lm_prob) / 2.0
                    
                    max_emotion = max(combined_probs, key=combined_probs.get)
                    max_confidence = combined_probs[max_emotion]
                    
                    # 更新对话历史
                    if robot_text:
                        self.conversation_history.append((robot_text, 1))
                    self.conversation_history.append((user_text, 0))
                    
                    # 保存当前概率用于下次比较
                    self.previous_gcn_probs = combined_probs.copy()
                    
                    return {
                        'emotion': max_emotion,
                        'confidence': max_confidence,
                        'probabilities': combined_probs,
                        'source': 'combined_gcn_lm',
                        'reason': f"大模型重新判断后综合DialogueGCN和大模型结果"
                    }
            else:
                print(f"🎯 置信度未显著增加，使用DialogueGCN结果: {gcn_emotion}")
                # 更新对话历史
                if robot_text:
                    self.conversation_history.append((robot_text, 1))
                self.conversation_history.append((user_text, 0))
                
                return {
                    'emotion': gcn_emotion,
                    'confidence': gcn_result['confidence'],
                    'probabilities': gcn_probs,
                    'source': 'dialoguegcn_no_increase',
                    'reason': f"置信度未显著增加，使用DialogueGCN的{gcn_emotion}"
                }
        
        # 更新对话历史
        if robot_text:
            self.conversation_history.append((robot_text, 1))
        self.conversation_history.append((user_text, 0))
        
        # 保持历史长度
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # 保存当前GCN概率用于下次比较
        self.previous_gcn_probs = gcn_probs.copy()
        
        return {
            'emotion': gcn_emotion,
            'confidence': gcn_result['confidence'],
            'probabilities': gcn_probs,
            'source': 'dialoguegcn_default',
            'reason': f"使用DialogueGCN默认结果: {gcn_emotion}"
        }
    
    def run_pure_dgcn_interactive(self):
        """纯DialogueGCN交互式评估模式
        
        只使用DialogueGCN进行预测，不使用大模型混合策略
        用户可以交互式输入文本，系统返回DialogueGCN的预测结果
        """
        print("\n" + "="*60)
        print("🎯 纯DialogueGCN交互式评估")
        print("="*60)
        print("只使用DialogueGCN模型进行情绪预测")
        print("不使用大模型混合策略")
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
                    # 输出统计信息
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
                    print("🗑️ 对话历史已清空")
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
                        print(f"最快/最慢: {np.min(prediction_times):.2f}ms / {np.max(prediction_times):.2f}ms")
                    print(f"对话历史长度: {len(self.conversation_history)}")
                    if emotion_counts:
                        print(f"情绪分布:")
                        for emotion, count in emotion_counts.most_common():
                            print(f"  {emotion}: {count} ({count/total_predictions*100:.1f}%)")
                    continue
                    
                elif not user_input:
                    print("⚠️ 请输入有效内容")
                    continue
                
                # 使用纯DialogueGCN预测
                print("\n🔍 正在使用DialogueGCN分析情绪...")
                t0 = time.perf_counter()
                gcn_result = self.get_dialoguegcn_prediction(user_input, robot_text="")
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                
                if gcn_result:
                    emotion = gcn_result['emotion']
                    confidence = gcn_result['confidence']
                    probabilities = gcn_result['probabilities']
                    
                    # 更新统计
                    total_predictions += 1
                    prediction_times.append(elapsed_ms)
                    emotion_counts[emotion] += 1
                    
                    # 显示结果
                    print(f"\n📊 DialogueGCN预测结果:")
                    print(f"   预测情绪: {emotion} (置信度: {confidence:.2%})")
                    print(f"   预测耗时: {elapsed_ms:.2f}ms")
                    print(f"   所有情绪概率:")
                    for emo, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                        bar = "█" * int(prob * 20)
                        print(f"     {emo:8}: {prob:.2%} {bar}")
                    
                    # 更新对话历史
                    self.conversation_history.append((user_input, 0))
                    
                else:
                    print("❌ DialogueGCN预测失败")
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
    
    def run_demo(self):
        """运行交互式demo"""
        print("\n" + "="*60)
        print("🎭 最终混合情绪识别系统")
        print("="*60)
        print("结合DialogueGCN和大模型的智能情绪判断")
        print("支持多轮对话，智能算法决策")
        print("输入 'quit' 退出程序")
        print("输入 'clear' 清空对话历史")
        print("输入 'reset' 重置对话上下文（等同于clear）")
        print("输入 'history' 查看对话历史")
        print("输入 'test' 运行测试用例（混合策略）")
        print("输入 'test_lm' 仅大模型单轮评估(无上下文)")
        print("输入 'test_dgcn' 纯DialogueGCN测试(文件评估)")
        print("输入 'interactive_dgcn' 纯DialogueGCN交互式评估")
        print("="*60)
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 用户: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 再见！")
                    break
                elif user_input.lower() == 'clear':
                    self.reset_dialogue_context()
                    print("🗑️ 对话历史已清空")
                    continue
                elif user_input.lower() == 'reset':
                    self.reset_dialogue_context()
                    print("🔄 对话上下文已重置")
                    continue
                elif user_input.lower() == 'history':
                    print("\n📜 对话历史:")
                    if not self.conversation_history:
                        print("   暂无对话历史")
                    else:
                        for i, (text, speaker) in enumerate(self.conversation_history[-5:]):
                            speaker_name = "用户" if speaker == 0 else "机器人"
                            print(f"   {i+1}. {speaker_name}: {text}")
                    continue
                elif user_input.lower() == 'test':
                    self.run_test_cases()
                    continue
                elif user_input.lower() == 'test_lm':
                    self.run_large_model_single_turn_test()
                    continue
                elif user_input.lower() == 'test_dgcn':
                    self.run_pure_dialoguegcn_test()
                    continue
                elif user_input.lower() == 'interactive_dgcn':
                    self.run_pure_dgcn_interactive()
                    continue
                elif not user_input:
                    print("⚠️ 请输入有效内容")
                    continue
                
                # 在用户输入后立即预测情绪
                print("\n🔍 正在分析用户情绪...")
                result = self.predict_emotion(user_input)
                
                if result:
                    print(f"\n📊 情绪分析结果:")
                    print(f"   预测情绪: {result['emotion']} (置信度: {result['confidence']:.2%})")
                    print(f"   判断来源: {result['source']}")
                    print(f"   判断依据: {result['reason']}")
                    print(f"   所有情绪概率:")
                    for emotion, prob in result['probabilities'].items():
                        bar = "█" * int(prob * 20)
                        print(f"     {emotion:8}: {prob:.2%} {bar}")
                else:
                    print("❌ 情绪分析失败")
                
                # 获取机器人回复（可选）
                robot_input = input("\n🤖 机器人: ").strip()
                if robot_input:
                    # 如果有机器人回复，更新对话历史
                    self.conversation_history.append((robot_input, 1))
                    print(f"✅ 机器人回复已记录: {robot_input}")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
                import traceback
                traceback.print_exc()
    
    def run_test_cases(self, data_path: str = 'improved_test_data.xlsx'):
        """基于test_data.xlsx运行评估，用数据文件中的标签对比预测结果。

        要求：
        - 数据列包含: dialogue_id, speaker(user/robot), utterance, emotion_label
        - 只有speaker为user时才有emotion_label
        - 每段对话都是机器人先说：需将机器人轮作为历史写入，不触发预测
        - 对每条用户(有标签)数据输出：utterance、耗时、预测与实际情绪标签
        - 最终输出：总体准确率、各类情绪召回率、平均耗时、耗时5/50/95分位
        - 额外统计：当最终结果来自DialogueGCN时的准确率
        """
        # 尝试导入sklearn以便计算召回率，不可用则回退到手工计算
        try:
            from sklearn.metrics import precision_recall_fscore_support
            _SK_AVAILABLE = True
        except Exception:
            _SK_AVAILABLE = False

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
        print("\n🧪 基于文件进行评估...")
        print(f"总记录数: {len(df)} | 用户(有标签)记录数: {len(user_rows)}")

        label_map = {
            'neutral': 'NEUTRAL',
            'happy': 'HAPPY',
            'sad': 'SAD',
            'angry': 'ANGRY',
            'surprise': 'SURPRISE'
        }
        valid_labels = ['NEUTRAL', 'ANGRY', 'SAD', 'HAPPY', 'SURPRISE']

        y_true, y_pred, times = [], [], []
        # 额外统计：记录每次预测的来源
        sources = []

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
                    # 机器人轮：检查是否有情绪标签，如果有则进行预测
                    if utterance.strip():
                        self.conversation_history.append((utterance.strip(), 1))
                    
                    # 如果robot有情绪标签，也进行预测
                    if true_label.strip():
                        t0 = time.perf_counter()
                        result = self.predict_emotion(utterance, robot_text="", dialogue_id=dialogue_id)
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0

                        pred = (result.get('emotion') or '').upper()
                        source = result.get('source', 'unknown')
                        mapped_true = label_map.get(true_label.lower(), true_label.upper())
                        y_true.append(mapped_true)
                        y_pred.append(pred)
                        times.append(elapsed_ms)
                        sources.append(source)

                        print(f"Robot话语: {utterance}")
                        print(f"耗时: {elapsed_ms:.1f}ms | 预测: {pred} | 实际: {mapped_true} | 来源: {source}")
                    continue

                if speaker == 'user':
                    # 用户轮：若无标签，则仅作为历史；有标签则进行预测比对
                    if not true_label.strip():
                        if utterance.strip():
                            self.conversation_history.append((utterance.strip(), 0))
                        continue

                    # 有标签用户轮：执行预测
                    t0 = time.perf_counter()
                    result = self.predict_emotion(utterance, robot_text="", dialogue_id=dialogue_id)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    pred = (result.get('emotion') or '').upper()
                    source = result.get('source', 'unknown')
                    mapped_true = label_map.get(true_label.lower(), true_label.upper())
                    y_true.append(mapped_true)
                    y_pred.append(pred)
                    times.append(elapsed_ms)
                    sources.append(source)

                    print(f"User话语: {utterance}")
                    print(f"耗时: {elapsed_ms:.1f}ms | 预测: {pred} | 实际: {mapped_true} | 来源: {source}")

        # 汇总指标
        if not y_true:
            print("\n❌ 没有可评估的标注数据（用户或机器人）")
            return

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        accuracy = float((y_true_np == y_pred_np).mean())

        print("\n—— 指标汇总 ——")
        print(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # 各类召回率
        print("各类情绪召回率:")
        if _SK_AVAILABLE:
            try:
                _, recall, _, support = precision_recall_fscore_support(
                    y_true_np, y_pred_np, labels=valid_labels, zero_division=0
                )
                for lbl, r, sup in zip(valid_labels, recall, support):
                    print(f"  {lbl:<9} Recall {r:.4f} (support={int(sup)})")
            except Exception as e:
                print(f"  计算召回率失败: {e}")
        else:
            # 手工计算召回率
            for lbl in valid_labels:
                sup = int((y_true_np == lbl).sum())
                tp = int(((y_true_np == lbl) & (y_pred_np == lbl)).sum())
                rec = tp / sup if sup > 0 else 0.0
                print(f"  {lbl:<9} Recall {rec:.4f} (support={sup})")

        # 耗时统计
        t = np.array(times, dtype=float)
        if t.size > 0:
            print("耗时统计:")
            print(f"  平均耗时: {t.mean():.2f}ms")
            print(f"  5分位: {np.percentile(t, 5):.2f}ms | 50分位: {np.percentile(t, 50):.2f}ms | 95分位: {np.percentile(t, 95):.2f}ms")
        else:
            print("耗时统计: 无")
        
        # 统计DialogueGCN结果的准确率
        print("\n—— DialogueGCN结果准确率 ——")
        # 找出所有来自DialogueGCN的预测
        dgcn_sources = ['dialoguegcn_non_neutral_stable', 'dialoguegcn_second_highest', 
                        'dialoguegcn_no_increase', 'dialoguegcn_default', 'dialoguegcn']
        dgcn_mask = np.array([s in dgcn_sources for s in sources])
        if dgcn_mask.sum() > 0:
            dgcn_y_true = y_true_np[dgcn_mask]
            dgcn_y_pred = y_pred_np[dgcn_mask]
            dgcn_accuracy = float((dgcn_y_true == dgcn_y_pred).mean())
            print(f"DialogueGCN结果数量: {dgcn_mask.sum()} / {len(sources)} ({dgcn_mask.sum()/len(sources)*100:.1f}%)")
            print(f"DialogueGCN结果准确率: {dgcn_accuracy:.4f} ({dgcn_accuracy*100:.2f}%)")
        else:
            print("无DialogueGCN结果")

    def run_large_model_single_turn_test(self, data_path: str = 'improved_test_data.xlsx'):
        """仅调用大模型（无上下文）对用户标注数据进行单轮评估。

        - 不使用对话历史与GCN，仅对每条用户(有标签)话语单独调用大模型。
        - 输出逐条结果（话语、耗时、预测、实际），并统计准确率、各类召回率、平均耗时、5/95分位。
        """
        # sklearn 可选
        try:
            from sklearn.metrics import precision_recall_fscore_support
            _SK_AVAILABLE = True
        except Exception:
            _SK_AVAILABLE = False

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

        # 仅取用户且有标签的数据
        user_df = df[(df['speaker'] == 'user') & df['emotion_label'].notna() & (df['emotion_label'].astype(str) != '')].copy()
        if user_df.empty:
            print("❌ 无用户标注数据可评估")
            return

        print("\n🧪 仅大模型单轮评估(无上下文)...")
        print(f"样本数: {len(user_df)}")

        label_map = {
            'neutral': 'NEUTRAL',
            'happy': 'HAPPY',
            'sad': 'SAD',
            'angry': 'ANGRY',
            'surprise': 'SURPRISE'
        }
        valid_labels = ['NEUTRAL', 'ANGRY', 'SAD', 'HAPPY', 'SURPRISE']

        y_true, y_pred, times = [], [], []
        for _, row in user_df.reset_index(drop=True).iterrows():
            utterance = '' if pd.isna(row['utterance']) else str(row['utterance'])
            true_raw = row['emotion_label']
            true_label = label_map.get(str(true_raw).lower(), str(true_raw).upper())

            # 调用大模型（无上下文）
            t0 = time.perf_counter()
            lm_result = self.large_model_client.analyze_emotion(utterance, context="")
            elapsed_ms_outer = (time.perf_counter() - t0) * 1000.0
            pred_label = (lm_result.get('emotion') or '').upper()
            elapsed_ms = float(lm_result.get('response_time_ms', elapsed_ms_outer))

            y_true.append(true_label)
            y_pred.append(pred_label)
            times.append(elapsed_ms)

            print(f"话语: {utterance}")
            print(f"耗时: {elapsed_ms:.1f}ms | 预测: {pred_label} | 实际: {true_label}")

        # 汇总
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        accuracy = float((y_true_np == y_pred_np).mean())

        print("\n—— 指标汇总（大模型单轮）——")
        print(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

        print("各类情绪召回率:")
        if _SK_AVAILABLE:
            try:
                _, recall, _, support = precision_recall_fscore_support(
                    y_true_np, y_pred_np, labels=valid_labels, zero_division=0
                )
                for lbl, r, sup in zip(valid_labels, recall, support):
                    print(f"  {lbl:<9} Recall {r:.4f} (support={int(sup)})")
            except Exception as e:
                print(f"  计算召回率失败: {e}")
        else:
            for lbl in valid_labels:
                sup = int((y_true_np == lbl).sum())
                tp = int(((y_true_np == lbl) & (y_pred_np == lbl)).sum())
                rec = tp / sup if sup > 0 else 0.0
                print(f"  {lbl:<9} Recall {rec:.4f} (support={sup})")

        t = np.array(times, dtype=float)
        if t.size > 0:
            print("耗时统计:")
            print(f"  平均耗时: {t.mean():.2f}ms")
            print(f"  5分位: {np.percentile(t, 5):.2f}ms | 50分位: {np.percentile(t, 50):.2f}ms | 95分位: {np.percentile(t, 95):.2f}ms")
        else:
            print("耗时统计: 无")
    
    def run_pure_dialoguegcn_test(self, data_path: str = 'improved_test_data.xlsx'):
        """纯DialogueGCN测试（不使用大模型混合策略）
        
        - 对每条用户话语仅使用DialogueGCN进行预测
        - 输出准确率、各类召回率、耗时统计（包括50分位）
        """
        try:
            from sklearn.metrics import precision_recall_fscore_support
            _SK_AVAILABLE = True
        except Exception:
            _SK_AVAILABLE = False

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

        df = df.copy()
        df['speaker'] = df['speaker'].astype(str)
        df['dialogue_id'] = df['dialogue_id']

        user_rows = df[(df['speaker'] == 'user') & df['emotion_label'].notna() & (df['emotion_label'].astype(str) != '')]
        robot_rows = df[(df['speaker'] == 'robot') & df['emotion_label'].notna() & (df['emotion_label'].astype(str) != '')]
        print("\n🧪 纯DialogueGCN测试（不使用混合策略）...")
        print(f"总记录数: {len(df)} | 用户(有标签)记录数: {len(user_rows)} | 机器人(有标签)记录数: {len(robot_rows)}")

        label_map = {
            'neutral': 'NEUTRAL',
            'happy': 'HAPPY',
            'sad': 'SAD',
            'angry': 'ANGRY',
            'surprise': 'SURPRISE'
        }
        valid_labels = ['NEUTRAL', 'ANGRY', 'SAD', 'HAPPY', 'SURPRISE']

        y_true, y_pred, times = [], [], []

        # 按对话分组
        for dialogue_id, group in df.groupby('dialogue_id', sort=False):
            print(f"\n=== 对话 {dialogue_id} 开始，{len(group)} 条 ===")
            # 重置内部状态
            self.reset_dialogue_context()

            group = group.reset_index(drop=True)
            for idx, row in group.iterrows():
                speaker = str(row['speaker']).strip().lower()
                utterance = str(row['utterance']) if not pd.isna(row['utterance']) else ''
                true_label_raw = row.get('emotion_label')
                true_label = '' if pd.isna(true_label_raw) else str(true_label_raw)

                if speaker == 'robot':
                    if utterance.strip():
                        self.conversation_history.append((utterance.strip(), 1))
                    
                    # 如果robot有情绪标签，也进行预测
                    if true_label.strip():
                        t0 = time.perf_counter()
                        gcn_result = self.get_dialoguegcn_prediction(utterance, robot_text="")
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0

                        if gcn_result:
                            pred = gcn_result['emotion']
                        else:
                            pred = 'NEUTRAL'  # 失败时默认NEUTRAL
                            print(f"⚠️ DialogueGCN预测失败，默认为NEUTRAL")

                        mapped_true = label_map.get(true_label.lower(), true_label.upper())
                        y_true.append(mapped_true)
                        y_pred.append(pred)
                        times.append(elapsed_ms)

                        print(f"Robot话语: {utterance}")
                        print(f"耗时: {elapsed_ms:.1f}ms | 预测: {pred} | 实际: {mapped_true}")
                    continue

                if speaker == 'user':
                    if not true_label.strip():
                        if utterance.strip():
                            self.conversation_history.append((utterance.strip(), 0))
                        continue

                    # 有标签用户轮：仅使用DialogueGCN预测
                    t0 = time.perf_counter()
                    gcn_result = self.get_dialoguegcn_prediction(utterance, robot_text="")
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    if gcn_result:
                        pred = gcn_result['emotion']
                        # 更新对话历史
                        self.conversation_history.append((utterance.strip(), 0))
                    else:
                        pred = 'NEUTRAL'  # 失败时默认NEUTRAL
                        print(f"⚠️ DialogueGCN预测失败，默认为NEUTRAL")

                    mapped_true = label_map.get(true_label.lower(), true_label.upper())
                    y_true.append(mapped_true)
                    y_pred.append(pred)
                    times.append(elapsed_ms)

                    print(f"User话语: {utterance}")
                    print(f"耗时: {elapsed_ms:.1f}ms | 预测: {pred} | 实际: {mapped_true}")

        # 汇总指标
        if not y_true:
            print("\n❌ 没有可评估的标注数据（用户或机器人）")
            return

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        accuracy = float((y_true_np == y_pred_np).mean())

        print("\n—— 纯DialogueGCN指标汇总 ——")
        print(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # 各类召回率
        print("各类情绪召回率:")
        if _SK_AVAILABLE:
            try:
                _, recall, _, support = precision_recall_fscore_support(
                    y_true_np, y_pred_np, labels=valid_labels, zero_division=0
                )
                for lbl, r, sup in zip(valid_labels, recall, support):
                    print(f"  {lbl:<9} Recall {r:.4f} (support={int(sup)})")
            except Exception as e:
                print(f"  计算召回率失败: {e}")
        else:
            for lbl in valid_labels:
                sup = int((y_true_np == lbl).sum())
                tp = int(((y_true_np == lbl) & (y_pred_np == lbl)).sum())
                rec = tp / sup if sup > 0 else 0.0
                print(f"  {lbl:<9} Recall {rec:.4f} (support={sup})")

        # 耗时统计
        t = np.array(times, dtype=float)
        if t.size > 0:
            print("耗时统计:")
            print(f"  平均耗时: {t.mean():.2f}ms")
            print(f"  5分位: {np.percentile(t, 5):.2f}ms | 50分位: {np.percentile(t, 50):.2f}ms | 95分位: {np.percentile(t, 95):.2f}ms")
        else:
            print("耗时统计: 无")

def main():
    """主函数"""
    # 查找可用的模型文件
    model_files = []
    for file in os.listdir('.'):
        if file.startswith('chinese_dialoguegcn_model_epoch_') and file.endswith('.pth'):
            model_files.append(file)
    
    if not model_files:
        print("❌ 未找到训练好的模型文件")
        print("请先运行训练脚本生成模型文件")
        return
    
    # 选择最新的模型
    latest_model = sorted(model_files)[-1]
    print(f"📁 找到模型文件: {latest_model}")
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ 使用设备: {device}")
    
    # 创建系统实例
    system = FinalEmotionSystem(latest_model, device)
    
    if system.model is None:
        print("❌ 系统初始化失败")
        return
    
    # 运行交互式demo
    system.run_demo()

if __name__ == "__main__":
    main()
 