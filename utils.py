import torch
import torch.nn.functional as F
import types

def sinkhorn_attention(query, key, value, attention_mask=None, epsilon=0.05, steps=3):
    """
    Sinkhorn Optimal Transport Attention
    """
    b, h, l_q, d = query.size()
    _, _, l_k, _ = key.size()

    # 1. Cost Matrix (1 - Cosine Similarity)
    q_norm = F.normalize(query, p=2, dim=-1)
    k_norm = F.normalize(key, p=2, dim=-1)
    cost = 1 - torch.matmul(q_norm, k_norm.transpose(-1, -2)) 

    # 2. Marginals
    # 文本端 (Query): 均匀分布
    nu = torch.ones(b, h, l_q, 1, device=query.device) / l_q
    # 视觉端 (Key): 均匀分布 (进阶可改为显著性加权)
    mu = torch.ones(b, h, l_k, 1, device=query.device) / l_k

    # 3. Sinkhorn Loop
    f = torch.zeros_like(nu)
    g = torch.zeros_like(mu)
    
    for _ in range(steps):
        tmp = g.transpose(-1, -2) - cost / epsilon
        f = -epsilon * torch.logsumexp(tmp, dim=-1, keepdim=True) + epsilon * torch.log(nu)
        tmp = f - cost / epsilon
        g = -epsilon * torch.logsumexp(tmp.transpose(-1, -2), dim=-1, keepdim=True) + epsilon * torch.log(mu)

    # 4. Transport Plan
    P = torch.exp((f + g.transpose(-1, -2) - cost) / epsilon)
    
    # 5. Apply Output
    return torch.matmul(P, value)

def gpt2_cross_attention_forward_proxy(
    self,
    hidden_states,
    layer_past=None,
    past_key_values=None,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    use_cache=False,
    output_attentions=False,
    **kwargs
):
    """
    Train-free OT-RAG Proxy (Log-Space Bias Version).
    修复 Mode Collapse 问题，使用对数域加法代替线性混合。
    """
    # ================= PARAMETERS =================
    # 调小 Alpha，从 0.05 开始尝试，不要一开始就 0.2
    OT_ALPHA = 0.05  
    # 调小 Epsilon，让 OT 矩阵更尖锐 (Sharper)
    OT_EPSILON = 0.05 
    # ==============================================

    # Helper: Split Heads
    def _my_split_heads(tensor, num_heads, head_dim):
        new_shape = tensor.size()[:-1] + (num_heads, head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    # Helper: Merge Heads
    def _my_merge_heads(tensor, num_heads, head_dim):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * head_dim,)
        return tensor.view(new_shape)

    # 1. 统一 Cache
    if past_key_values is None and layer_past is not None:
        past_key_values = layer_past

    # 2. 获取 Key, Value
    key, value = None, None
    if past_key_values is not None:
        possible_key = past_key_values[0]
        if isinstance(possible_key, tuple):
            key, value = possible_key[0], possible_key[1]
        else:
            key, value = past_key_values[0], past_key_values[1]
        while isinstance(key, tuple): key = key[0]
        while isinstance(value, tuple): value = value[0]
    else:
        if encoder_hidden_states is None:
             raise ValueError("OT-RAG Error")
        key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        key = _my_split_heads(key, self.num_heads, self.head_dim)
        value = _my_split_heads(value, self.num_heads, self.head_dim)

    # 3. 获取 Query
    query = self.q_attn(hidden_states)
    query = _my_split_heads(query, self.num_heads, self.head_dim)

    # ------------------------------------------------------------------
    # [核心修改] Log-Space Fusion
    # ------------------------------------------------------------------
    
    # A. 计算原始 Attention Scores (Logits)
    # Shape: [Batch, Head, Q_Len, K_Len]
    attn_scores = torch.matmul(query, key.transpose(-1, -2))
    attn_scores = attn_scores / (float(value.size(-1)) ** 0.5)
    
    # B. 计算 OT Plan (Probabilities)
    ot_plan = sinkhorn_attention_weights_only(query, key, epsilon=OT_EPSILON, steps=3)
    
    # [DEBUG 打印] - 第一次运行时取消注释，观察 OT 是否是均匀分布
    # if torch.rand(1).item() < 0.01: # 抽样打印
    #     print(f"OT Max: {ot_plan.max().item():.4f}, Min: {ot_plan.min().item():.4f}, Mean: {ot_plan.mean().item():.4f}")
    #     # 如果 Max 和 Min 非常接近 (例如都是 0.002)，说明 OT 失效了，变成了均匀噪声

    # C. 将 OT 转换为 Bias (Log domain) 并加到原始 Score 上
    # 逻辑：我们将 OT 视为一种“先验偏置”。如果 OT 认为某处概率高，就给它的 Score 加分。
    # log(ot_plan + 1e-9) 会将概率转回 logit 空间
    ot_bias = torch.log(ot_plan + 1e-9)
    
    # 融合：Standard Scores + alpha * OT_Bias
    # 这样保留了 Softmax 的非线性竞争机制
    final_scores = attn_scores + OT_ALPHA * ot_bias

    # D. Softmax
    if attention_mask is not None:
        final_scores = final_scores + attention_mask
        
    attn_weights = F.softmax(final_scores, dim=-1)
    
    # Dropout
    attn_weights = self.attn_dropout(attn_weights)

    # 4. 计算输出
    attn_output = torch.matmul(attn_weights, value)

    # 5. 投影返回
    attn_output = _my_merge_heads(attn_output, self.num_heads, self.head_dim)
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, None)
    return outputs

# 保持 sinkhorn 函数不变，或者微调 epsilon
def sinkhorn_attention_weights_only(query, key, epsilon=0.05, steps=100):
    b, h, l_q, d = query.size()
    _, _, l_k, _ = key.size()

    q_norm = F.normalize(query, p=2, dim=-1)
    k_norm = F.normalize(key, p=2, dim=-1)
    
    # Cost = 1 - Cosine. 
    # 放大 Cost 可以让分布更尖锐。尝试乘一个 scaling factor
    cost = 1 - torch.matmul(q_norm, k_norm.transpose(-1, -2)) 
    
    # Marginals
    nu = torch.ones(b, h, l_q, 1, device=query.device) / l_q
    mu = torch.ones(b, h, l_k, 1, device=query.device) / l_k

    f = torch.zeros_like(nu)
    g = torch.zeros_like(mu)
    
    for _ in range(steps):
        tmp = g.transpose(-1, -2) - cost / epsilon
        f = -epsilon * torch.logsumexp(tmp, dim=-1, keepdim=True) + epsilon * torch.log(nu)
        tmp = f - cost / epsilon
        g = -epsilon * torch.logsumexp(tmp.transpose(-1, -2), dim=-1, keepdim=True) + epsilon * torch.log(mu)

    P = torch.exp((f + g.transpose(-1, -2) - cost) / epsilon)
    return P

def apply_ot_rag_patch(model):
    print(">> [OT-RAG] Injecting Optimal Transport alignment (Forward Patch)...")
    count = 0
    
    for name, module in model.named_modules():
        # 核心判断：只修改 Cross Attention 层
        if hasattr(module, "is_cross_attention") and module.is_cross_attention:
            
            # 1. 检查必要组件是否存在 (GPT2 结构特征)
            if not hasattr(module, 'q_attn') or not hasattr(module, 'c_attn'):
                print(f"   !! Skipped {name}: Missing q_attn/c_attn layers.")
                continue

            # 2. 绑定新方法到实例上 (Monkey Patch)
            # 使用 types.MethodType 将我们的函数绑定为该实例的方法
            module.forward = types.MethodType(gpt2_cross_attention_forward_proxy, module)
            
            count += 1
            print(f"   -> Patched Cross-Attention layer: {name}")

    if count == 0:
        print("!! [Warning] No Cross-Attention layers found to patch.")
    else:
        print(f">> [OT-RAG] Successfully patched {count} layers.")