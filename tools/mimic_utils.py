from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticOTProjection(nn.Module):
    def __init__(self, visual_dim, text_dim, num_anchors=14, epsilon=0.05, iter=3):
        """
        基于语义锚点的 OT 投影层
        Args:
            visual_dim: 输入视觉特征维度 (Dv)
            text_dim:   目标文本特征维度 (Dtext, 也是语义锚点的维度)
            num_anchors: 语义锚点数量 (K, 对应疾病概念数量)
            epsilon:    熵正则化系数 (初始值)
            iter:       Sinkhorn 迭代次数
        """
        super().__init__()
        self.num_anchors = num_anchors
        self.text_dim = text_dim
        self.iter = iter

        # 1. 语义锚点 (Semantic Anchors / Prototypes) P
        # Shape: [1, K, D_text]
        # 初始化：默认为随机，但提供 load_pretrained_prototypes 方法以支持 "文本主导性"
        self.prototypes = nn.Parameter(torch.randn(1, num_anchors, text_dim))
        
        # 2. 视觉适配层 (W_v)
        # 仅用于计算 Cost 时的空间对齐：Visual Space -> Text Space
        # 这就是理论中的 W_v，用于计算 Cosine 距离
        self.visual_adapter = nn.Linear(visual_dim, text_dim)

        # 3. 可学习的熵正则化系数 (Learnable Temperature)
        # 使用 Softplus 确保 epsilon 始终 > 0
        self.epsilon_param = nn.Parameter(torch.tensor(float(epsilon)))

    def load_pretrained_prototypes(self, embeddings):
        """
        加载预训练的文本 Embeddings (如医学词典聚类中心)
        embeddings: [num_anchors, text_dim]
        """
        if embeddings.shape != (self.num_anchors, self.text_dim):
            raise ValueError(f"Shape mismatch: expected ({self.num_anchors}, {self.text_dim}), "
                             f"got {embeddings.shape}")
        with torch.no_grad():
            self.prototypes.data.copy_(embeddings.unsqueeze(0))
        print("Pretrained semantic anchors loaded.")

    def get_epsilon(self):
        # 保证 epsilon 为正数，且不过小导致数值不稳定
        return F.softplus(self.epsilon_param) + 1e-4

    def sinkhorn(self, x, y):
        # x: [Batch, N, D_text] (适配后的视觉特征)
        # y: [Batch, K, D_text] (语义锚点)
        
        epsilon = self.get_epsilon()

        # --- A. 计算 Cost Matrix (语义相关性度量) ---
        # 使用余弦距离: C_ij = 1 - cos(v_i, p_j)
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        # [Batch, N, K]
        C = 1 - torch.bmm(x_norm, y_norm.transpose(1, 2)) 

        # --- B. 初始化对偶变量 (Log domain) ---
        B, N, K = C.shape
        mu = torch.zeros(B, N, 1, device=x.device)
        nu = torch.zeros(B, 1, K, device=x.device)

        # --- C. Sinkhorn 迭代 ---
        for _ in range(self.iter):
            # Update mu (Visual side)
            mu = -epsilon * torch.logsumexp((nu - C) / epsilon, dim=2, keepdim=True)
            # Update nu (Semantic Anchor side)
            nu = -epsilon * torch.logsumexp((mu - C) / epsilon, dim=1, keepdim=True)

        # --- D. 计算传输方案 T ---
        # T_ij 代表：第 i 个视觉 patch 应该分配多少“质量”给第 j 个语义锚点
        log_T = (mu + nu - C) / epsilon
        T = torch.exp(log_T) # [Batch, N, K]
        
        return T

    def forward(self, x):
        """
        x: [Batch, N, Visual_Dim] (原始视觉特征)
        Returns:
           projected_feats: [Batch, N, Text_Dim] (重构后的特征)
        """
        batch_size = x.shape[0]

        # 1. 空间变换与对齐 (Visual -> Text Space)
        # v_i * W_v
        x_adapted = self.visual_adapter(x) # [Batch, N, Text_Dim]
        
        # 2. 准备语义锚点
        # P
        anchors = self.prototypes.expand(batch_size, -1, -1) # [Batch, K, Text_Dim]
        
        # 3. 计算传输矩阵 T
        # T: [Batch, N, K]
        T = self.sinkhorn(x_adapted, anchors)
        
        # 4. 重心映射 (Barycentric Mapping)
        # 理论核心：Output = P 的加权组合
        # 我们需要对 T 进行行归一化，确保每个视觉 Patch 的权重和为 1
        # 这样生成的特征才是语义锚点的凸组合 (Convex Combination)
        T_norm = T / (T.sum(dim=2, keepdim=True) + 1e-8)
        
        # [Batch, N, K] x [Batch, K, Text_Dim] -> [Batch, N, Text_Dim]
        # 物理含义：每个位置 N 现在的特征，是由 K 个语义锚点根据 T 的权重混合而成的
        projected_feats = torch.bmm(T_norm, anchors)
        
        # 返回字典格式以兼容原有代码
        return {'projected_encoder_last_hidden_state': projected_feats}

def get_disease_embeddings(disease_list, model_name_or_path="distilgpt2", device="cpu"):
    """
    提取疾病名称的语义向量
    """
    print(f"Loading text encoder ({model_name_or_path}) to generate anchors...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # 使用基础模型提取特征 (不带 LM Head)
    text_model = AutoModel.from_pretrained(model_name_or_path).to(device)
    text_model.eval()

    embeddings = []
    
    # 也可以尝试 prompt engineering，比如 "Findings of {disease}"，
    # 但最简单的就是直接用单词，效果通常已经很好。
    
    with torch.no_grad():
        for disease in disease_list:
            # 1. Tokenize
            # add_prefix_space=True 通常对 GPT 类模型比较友好
            inputs = tokenizer(disease, return_tensors="pt").to(device)
            
            # 2. Forward Pass
            outputs = text_model(**inputs)
            
            # 3. 获取特征 (Last Hidden State)
            # Shape: [1, Sequence_Length, Hidden_Dim] (例如 [1, 3, 768])
            last_hidden_state = outputs.last_hidden_state
            
            # 4. Pooling (聚合)
            # 因为一个词可能被拆成多个 token (如 "Pneum" + "onia")
            # 或者由两个词组成 (如 "Pleural" + "Thickening")
            # 我们取平均值代表整个概念的中心
            concept_embedding = last_hidden_state.mean(dim=1).squeeze(0) # -> [768]
            
            embeddings.append(concept_embedding)

    # 堆叠成矩阵 [14, 768]
    anchor_matrix = torch.stack(embeddings)
    return anchor_matrix