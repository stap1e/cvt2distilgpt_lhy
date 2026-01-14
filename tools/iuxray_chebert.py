import json
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import BertConfig, BertModel, BertTokenizer
from collections import Counter, defaultdict
from tqdm import tqdm 

# ==========================================
# 1. 你的 CheXbert 类定义 (保持不变)
# ==========================================
class CheXbert(nn.Module):
    def __init__(self, ckpt_dir, bert_path, checkpoint_path, device, p=0.1):
        super(CheXbert, self).__init__()

        self.device = device
        
        # 这里的路径组合逻辑稍微调整了一下，确保兼容绝对路径和相对路径
        config_path = os.path.join(ckpt_dir, bert_path) if not os.path.isabs(bert_path) else bert_path
        
        self.tokenizer = BertTokenizer.from_pretrained(config_path)
        config = BertConfig().from_pretrained(config_path, local_files_only=True)

        with torch.no_grad():
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(p)
            hidden_size = self.bert.pooler.dense.in_features

            # Heads 0-12: 4 classes (Present, Absent, Unknown, Blank)
            self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
            # Head 13: 2 classes (Yes, No) for 'No Finding'
            self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

            # Load Checkpoint
            full_ckpt_path = os.path.join(ckpt_dir, checkpoint_path)
            print(f"Loading CheXbert weights from: {full_ckpt_path}")
            state_dict = torch.load(full_ckpt_path, map_location=device)['model_state_dict']

            new_state_dict = OrderedDict()
            new_state_dict["bert.embeddings.position_ids"] = torch.arange(config.max_position_embeddings).expand((1, -1))
            for key, value in state_dict.items():
                if 'bert' in key:
                    new_key = key.replace('module.bert.', 'bert.')
                elif 'linear_heads' in key:
                    new_key = key.replace('module.linear_heads.', 'linear_heads.')
                else:
                    new_key = key
                new_state_dict[new_key] = value

            self.load_state_dict(new_state_dict, strict=False)

        self.eval()
        self.to(device)

    def forward(self, reports):
        # 预处理
        processed_reports = []
        for r in reports:
            if not isinstance(r, str): r = "" # 防止 None
            r = r.strip().replace("\n", " ").replace("\s+", " ")
            processed_reports.append(r)
        
        if not processed_reports:
            return None

        with torch.no_grad():
            tokenized = self.tokenizer(processed_reports, padding='longest', truncation=True, max_length=512, return_tensors="pt")
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            last_hidden_state = self.bert(**tokenized)[0]
            cls = last_hidden_state[:, 0, :]
            cls = self.dropout(cls)

            predictions = []
            for i in range(14):
                predictions.append(self.linear_heads[i](cls).argmax(dim=1))

        return torch.stack(predictions, dim=1) # Shape: (Batch, 14)

# ==========================================
# 2. 标签转换逻辑 (Raw Class -> 0/1)
# ==========================================
def convert_chexbert_output_to_binary(preds, policy='u_zeros'):
    """
    将 CheXbert 的原始输出转换为 0/1 编码。
    Raw Classes: 
       0: Blank (Unmentioned)
       1: Positive (Present)
       2: Negative (Absent)
       3: Uncertain
    
    'No Finding' (Index 13) is special: 0=Positive, 1=Negative (通常 CheXbert 逻辑如此，需根据你的权重确认)
    但通常我们只需要前 13 个病理 + No Finding。
    """
    # 假设 preds shape: (Batch, 14)
    # 这里的映射逻辑基于常见 CheXbert 实现
    # Map: 1(Pos) -> 1, 3(Uncertain) -> 1/0, Others -> 0
    
    binary_preds = torch.zeros_like(preds, dtype=torch.float32)
    
    # 遍历前13个类别 (常规病理)
    # 通常映射: 1=Present -> 1; 3=Uncertain -> depends on policy; 0=Blank, 2=Absent -> 0
    positive_mask = (preds[:, :13] == 1)
    uncertain_mask = (preds[:, :13] == 3)
    
    binary_preds[:, :13][positive_mask] = 1.0
    
    if policy == 'u_ones':
        binary_preds[:, :13][uncertain_mask] = 1.0
    else:
        binary_preds[:, :13][uncertain_mask] = 0.0
        
    # 处理第 14 个 'No Finding'
    # 注意：CheXbert 原始逻辑 No Finding 有时是二分类 (0/1)。
    # 如果你的权重里 No Finding 是 0=Yes, 1=No，需要反转一下。
    # 这里假设 No Finding 如果被激活(比如是 0)，则设为 1。这里简单处理，直接保留原值或根据实际情况调整。
    # 为简单起见，通常主要关注前 13 个病理。如果需要 No Finding，通常是 0 位。
    # 这里我们直接将最后一列视为 No Finding 的 Positive 状态
    
    # 简单修正：如果前13个全是0，强制设 No Finding 为 1 (这也是一种常用策略)
    no_disease = (binary_preds[:, :13].sum(dim=1) == 0)
    binary_preds[no_disease, 13] = 1.0
    
    return binary_preds

# ==========================================
# 3. 数据处理主流程 (包含 Batch Inference)
# ==========================================
def process_json_with_chexbert(json_path, model, target_split='train', batch_size=32):
    
    # --- A. 读取并筛选数据 ---
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    raw_items = []
    
    # 智能解析结构 (复用之前的逻辑)
    if isinstance(data, list):
        data_src = data
    elif isinstance(data, dict):
        if 'images' in data: data_src = data['images']
        elif target_split in data: data_src = data[target_split]
        else: data_src = list(data.values())
    else:
        raise ValueError("Unknown JSON format")

    # 筛选 split 并提取 Report
    valid_samples = []
    for item in data_src:
        if not isinstance(item, dict): continue
        
        # 兼容处理
        curr_split = item.get('split', target_split) # 如果没有split字段，默认视为当前target
        
        if curr_split == target_split:
            report = item.get('report', "")
            if report: # 只处理有报告的数据
                valid_samples.append({
                    'id': item.get('id', 'unknown'),
                    'report': report
                })

    print(f"Split '{target_split}' 筛选出 {len(valid_samples)} 条有效样本。开始 CheXbert 推理...")

    # --- B. 批量推理 (Batch Inference) ---
    results = []
    
    # 使用 tqdm 显示进度
    for i in tqdm(range(0, len(valid_samples), batch_size), desc="Inferencing"):
        batch_items = valid_samples[i : i + batch_size]
        batch_reports = [item['report'] for item in batch_items]
        
        # 1. 模型推理
        raw_preds = model(batch_reports) # (Batch, 14)
        
        # 2. 转为 0/1 编码
        binary_labels = convert_chexbert_output_to_binary(raw_preds, policy='u_zeros')
        
        # 3. 存回结果
        for idx, item in enumerate(batch_items):
            results.append({
                'id': item['id'],
                'report': item['report'],
                'labels': binary_labels[idx].cpu().tolist() # 转为 list 方便存储/打印
            })

    return results

def group_and_analyze_labels(data_list, times=40):
    """
    根据 labels 对数据进行分组，并统计每组数量。
    
    Args:
        data_list: 包含 {'id':..., 'labels': [0,1...]} 的列表
    Returns:
        grouped_data: 字典 { label_tuple: [id1, id2, ...] }
    """
    # 1. 使用 defaultdict 方便存储
    # Key: 标签元组 (例如 (0, 0, 1, ...))
    # Value: 属于该组的样本 ID 列表
    grouped_data = defaultdict(list)
    
    print("正在进行分组统计...")
    
    for item in data_list:
        # 【关键步骤】将 list 转为 tuple，因为 list 不能做字典的 Key
        # 同时转为 int 看起来更整洁 (1.0 -> 1)
        label_key = tuple(int(x) for x in item['labels'])
        
        grouped_data[label_key].append(item['id'])
        
    # 2. 统计结果
    total_groups = len(grouped_data)
    
    print(f"\n====== 分组统计结果 ======")
    print(f"总共有 {total_groups} 种不同的标签组合 (Unique Label Patterns)。")
    print(f"--------------------------------")
    
    # 3. 按数量排序，打印前 20 个大组 (最常见的病理组合)
    # 按照 list 的长度 (即该组样本数) 降序排序
    sorted_groups = sorted(grouped_data.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (label_pattern, ids) in enumerate(sorted_groups):
        count = len(ids)
        if i < times: 
            print(f"Group {i+1}: Count = {count} | Label = {label_pattern}")
            # print(f"    Sample IDs (前3个): {ids[:3]}...") # 可选：查看部分ID
        
    return grouped_data

# ==========================================
# 4. 配置与运行
# ==========================================
if __name__ == "__main__":
    # --- 配置区域 ---
    # 请修改为你的实际路径
    CKPT_DIR = "/mnt/data/liuhongyu/rg/checkpoints/chexbert_weights" # 存放 bert_config.json 和 .pt 的目录
    BERT_PATH = "/mnt/data/liuhongyu/rg/checkpoints/bert-base-uncased" # 或者绝对路径 "/path/to/bert-base-uncased"
    CHECKPOINT_FILE = "/mnt/data/liuhongyu/rg/checkpoints/stanford/chexbert/chexbert.pth" # 你的 chexbert 权重文件名
    JSON_PATH = "/mnt/data/liuhongyu/IUXRay/hergen_iuxray.json"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. 初始化模型 ---
    print(f"Initializing CheXbert on {DEVICE}...")
    try:
        # 确保目录存在
        if not os.path.exists(CKPT_DIR):
             # 如果没有本地权重，这里演示如何从 huggingface 自动下载基础 bert
             # 你需要确保 ckpt_dir 指向包含 chexbert.pth 的位置
             print(f"Warning: {CKPT_DIR} not found. Please set correct path.")

        chexbert = CheXbert(
            ckpt_dir=CKPT_DIR,
            bert_path=BERT_PATH,
            checkpoint_path=CHECKPOINT_FILE,
            device=DEVICE
        )
        
        # --- 2. 处理数据 ---
        train_data = process_json_with_chexbert(
            json_path=JSON_PATH, 
            model=chexbert, 
            target_split='train', 
            batch_size=32 # 根据显存调整，32/64 通常没问题
        )

        # --- 3. 结果展示 ---
        times = 20
        print(f"\n--- 结果预览 (前{times}条) ---")
        for sample in train_data[:times]:
            print(f"ID: {sample['id']}")
            print(f"Labels: {sample['labels']}")
            print(f"report: {sample['report']}")
            print("-" * 30)
        
        if 'train_data' in locals() and train_data:
            label_groups = group_and_analyze_labels(train_data, 50)
            
    except Exception as e:
        print(f"\n运行出错: {e}")
        print("请检查 CKPT_DIR 和 CHECKPOINT_FILE 路径是否正确。")