import torch
import torch.nn as nn

class UOTLoss(nn.Module):
    def __init__(self, rho=0.1, epsilon=0.1, n_iters=50):
        """
        Args:
            rho: 边缘松弛项的惩罚系数 (控制 Unbalanced 的程度，越大越接近 Balanced OT)
            epsilon: 熵正则化系数 (控制平滑度)
            n_iters: Sinkhorn 迭代次数
        """
        super().__init__()
        self.rho = rho
        self.epsilon = epsilon
        self.n_iters = n_iters

    def forward(self, x, y, y_mask=None):
        """
        计算 Batch 内每对样本的 UOT 距离。
        Args:
            x: 图像特征 (B, N, D)  [N=patches]
            y: 文本特征 (B, M, D)  [M=tokens]
            y_mask: 文本掩码 (B, M), 真实token为1, pad为0
        """
        B, N, D = x.shape
        _, M, _ = y.shape
        
        # 1. 计算 Cost Matrix (Cosine Distance 或 Euclidean)
        # 这里使用 Cosine Distance: 1 - cos_sim
        x_norm = torch.nn.functional.normalize(x, dim=-1)
        y_norm = torch.nn.functional.normalize(y, dim=-1)
        # (B, N, M)
        cost_matrix = 1 - torch.bmm(x_norm, y_norm.transpose(1, 2))
        
        # 2. 初始化边缘分布 (Marginals)
        # 图像端：假设均匀分布
        mu = torch.ones(B, N, 1, device=x.device) / N
        
        # 文本端：考虑 Mask，只对非 Pad 区域分配质量
        if y_mask is not None:
            # (B, M, 1)
            nu = y_mask.unsqueeze(-1) 
            # 归一化，使得有效 Token 的概率和为 1
            nu = nu / (nu.sum(dim=1, keepdim=True) + 1e-8)
        else:
            nu = torch.ones(B, M, 1, device=x.device) / M

        # 3. Sinkhorn 算法 (Log-domain 稳定性更好，但这里用 Scaling-domain 简化版)
        # 预计算 Gibbs Kernel
        K = torch.exp(-cost_matrix / self.epsilon)
        
        # 迭代变量
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        
        # Sinkhorn-Knopp Iterations for Unbalanced OT
        # 这里的迭代公式针对 Unbalanced OT (KL-divergence penalty)
        fi = self.rho / (self.rho + self.epsilon)
        
        for _ in range(self.n_iters):
            # Update u
            # u = (mu / (K @ v)) ^ fi
            Kv = torch.bmm(K, v) + 1e-8
            u = (mu / Kv) ** fi
            
            # Update v
            # v = (nu / (K.T @ u)) ^ fi
            Ktu = torch.bmm(K.transpose(1, 2), u) + 1e-8
            v = (nu / Ktu) ** fi

        # 4. 计算 Transport Plan
        # P = u * K * v
        P = u * K * v.transpose(1, 2) # (B, N, M)
        
        # 5. 计算 Loss = sum(P * C)
        loss = torch.sum(P * cost_matrix, dim=(1, 2)) # (B,)
        
        return loss.mean()
    


class SelKDLoss(nn.Module):
    def __init__(self, rho=1.0, epsilon=0.1, n_iters=5):
        """
        rho: 边缘松弛系数。
             较小(如 1.0) -> 允许文本不看图 (Open-set/Dustbin 效果)。
             较大(如 100) -> 强制文本必须看图 (Closed-set)。
        epsilon: 熵正则。控制 Teacher 注意力的稀疏度 (越小越尖锐)。
        """
        super().__init__()
        self.rho = rho
        self.epsilon = epsilon
        self.n_iters = n_iters

    @torch.no_grad() # Teacher (OT) 计算不需要梯度
    def _compute_teacher_plan(self, x, y, y_mask):
        B, N, D = x.shape
        _, M, _ = y.shape

        # 1. Cost Matrix (1 - Cosine Similarity)
        x_norm = torch.nn.functional.normalize(x, dim=-1)
        y_norm = torch.nn.functional.normalize(y, dim=-1)
        cost_matrix = 1 - torch.bmm(y_norm, x_norm.transpose(1, 2)) # Shape: (B, M, N) [Text x Image]

        # 2. Marginals (先验分布)
        # 图像端 (x): 均匀分布 (假设每个 patch 重要性均等)
        mu = torch.ones(B, 1, N, device=x.device) / N
        # 文本端 (y): 根据 Mask 设定，Padding 的权重为 0
        if y_mask is not None:
            nu = y_mask.unsqueeze(-1) # (B, M, 1)
            nu = nu / (nu.sum(dim=1, keepdim=True) + 1e-8) # 归一化有效 Token
            nu = nu.transpose(1, 2) # (B, 1, M)
        else:
            nu = torch.ones(B, 1, M, device=x.device) / M

        # 3. Sinkhorn-Knopp (UOT 版本)
        K = torch.exp(-cost_matrix / self.epsilon)
        u = torch.ones_like(nu) # [修正] 初始化为 1，原代码为 0 会导致 nan
        v = torch.ones_like(mu) # [修正] 初始化为 1

        fi = self.rho / (self.rho + self.epsilon)

        for _ in range(self.n_iters):
            # 迭代公式 (注意维度匹配 B, M, N)
            # Update u (Text side)
            Kv = torch.bmm(K, v.transpose(1, 2)) # (B, M, 1)
            u = (nu.transpose(1, 2) / (Kv + 1e-8)) ** fi # (B, M, 1)
            
            # Update v (Image side)
            Ktu = torch.bmm(K.transpose(1, 2), u) # (B, N, 1)
            v = (mu.transpose(1, 2) / (Ktu + 1e-8)) ** fi # (B, N, 1)

        # 4. 得到 Teacher Plan
        # P = diag(u) * K * diag(v)
        P = u * K * v.transpose(1, 2) # (B, M, N)
        
        # [关键] 归一化行和，使其成为合法的 Attention 分布 (Sum=1)
        # 这样才能指导 Softmax 后的 Student Attention
        P_normalized = P / (P.sum(dim=-1, keepdim=True) + 1e-8)
        return P_normalized

    def forward(self, student_attn, visual_feats, text_feats, text_mask=None):
        """
        Args:
            student_attn: (B, Heads, M, N) 你的模型输出的 Cross-Attention 权重 (已 Softmax)
            visual_feats: (B, N, D) 图像特征
            text_feats:   (B, M, D) 文本特征 (Decoder Output)
            text_mask:    (B, M) padding mask
        """
        # 1. 计算 Teacher (OT Plan) - 基于当前的特征相似度
        teacher_plan = self._compute_teacher_plan(visual_feats, text_feats, text_mask) # (B, M, N)

        # 2. 对齐 Loss (KL Divergence / Cross Entropy)
        # 你的 student_attn 可能是多头的 (B, H, M, N)，我们让每个头都去学 Teacher
        # 或者你可以取头的平均：student_attn = student_attn.mean(dim=1)
        
        # 扩展 Teacher 维度以匹配 Head (B, 1, M, N)
        teacher_target = teacher_plan.unsqueeze(1) 
        
        # 为了数值稳定，使用 Cross Entropy 形式: - sum(P_teacher * log(P_student))
        # 加上 1e-8 防止 log(0)
        loss = -torch.sum(teacher_target * torch.log(student_attn + 1e-8), dim=-1) # Sum over Image dim

        # Apply Mask (Padding 处不计算 Loss)
        if text_mask is not None:
            loss = loss * text_mask.unsqueeze(1) # (B, H, M)

        return loss.mean()