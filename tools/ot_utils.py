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