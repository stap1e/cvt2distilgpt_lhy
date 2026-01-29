import json
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import BertConfig, BertModel, BertTokenizer
from collections import Counter, defaultdict
from tqdm import tqdm 

# ==========================================
# 1. CheXbert 类定义 (保持不变)
# ==========================================
class CheXbert(nn.Module):
    def __init__(self, ckpt_dir, bert_path, checkpoint_path, device, p=0.1):
        super(CheXbert, self).__init__()
        self.device = device
        config_path = os.path.join(ckpt_dir, bert_path) if not os.path.isabs(bert_path) else bert_path
        self.tokenizer = BertTokenizer.from_pretrained(config_path)
        config = BertConfig().from_pretrained(config_path, local_files_only=True)

        with torch.no_grad():
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(p)
            hidden_size = self.bert.pooler.dense.in_features
            # Heads 0-12: 4 classes; Head 13: 2 classes
            self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
            self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

            full_ckpt_path = os.path.join(ckpt_dir, checkpoint_path)
            print(f"Loading CheXbert weights from: {full_ckpt_path}")
            state_dict = torch.load(full_ckpt_path, map_location=device)['model_state_dict']
            new_state_dict = OrderedDict()
            new_state_dict["bert.embeddings.position_ids"] = torch.arange(config.max_position_embeddings).expand((1, -1))
            for key, value in state_dict.items():
                if 'bert' in key: new_key = key.replace('module.bert.', 'bert.')
                elif 'linear_heads' in key: new_key = key.replace('module.linear_heads.', 'linear_heads.')
                else: new_key = key
                new_state_dict[new_key] = value
            self.load_state_dict(new_state_dict, strict=False)

        self.eval()
        self.to(device)

    def forward(self, reports):
        processed_reports = []
        for r in reports:
            if not isinstance(r, str): r = ""
            r = r.strip().replace("\n", " ").replace("\s+", " ")
            processed_reports.append(r)
        
        if not processed_reports: return None

        with torch.no_grad():
            tokenized = self.tokenizer(processed_reports, padding='longest', truncation=True, max_length=512, return_tensors="pt")
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            last_hidden_state = self.bert(**tokenized)[0]
            cls = self.dropout(last_hidden_state[:, 0, :])
            predictions = []
            for i in range(14):
                predictions.append(self.linear_heads[i](cls).argmax(dim=1))
        return torch.stack(predictions, dim=1) 

# ==========================================
# 2. (已移除/跳过) 标签转换逻辑
# ==========================================
# def convert_chexbert_output_to_binary(...):
#     我们不再需要这个函数，因为我们要保留原始的 0,1,2,3 状态

# ==========================================
# 3. 数据处理主流程 (修改点：直接保存原始预测)
# ==========================================
def process_json_with_chexbert(json_path, model, target_split='train', batch_size=32):
    
    # --- A. 读取并筛选数据 (保持不变) ---
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list): data_src = data
    elif isinstance(data, dict):
        if 'images' in data: data_src = data['images']
        elif target_split in data: data_src = data[target_split]
        else: data_src = list(data.values())
    else: raise ValueError("Unknown JSON format")

    valid_samples = []
    for item in data_src:
        if not isinstance(item, dict): continue
        curr_split = item.get('split', target_split)
        if curr_split == target_split:
            report = item.get('report', "")
            if report:
                valid_samples.append({'id': item.get('id', 'unknown'), 'report': report})

    print(f"Split '{target_split}' 筛选出 {len(valid_samples)} 条有效样本。开始提取 4分类真值标签...")

    # --- B. 批量推理 ---
    results = []
    for i in tqdm(range(0, len(valid_samples), batch_size), desc="Inferencing"):
        batch_items = valid_samples[i : i + batch_size]
        batch_reports = [item['report'] for item in batch_items]
        
        # 1. 模型推理 (得到 0,1,2,3)
        raw_preds = model(batch_reports) 
        
        # 2. 【修改点】不再进行 binary 转换，直接转为 list
        # 0:Unmentioned, 1:Positive, 2:Negative, 3:Uncertain
        four_class_labels = raw_preds.cpu().tolist()
        
        # 3. 存回结果
        for idx, item in enumerate(batch_items):
            results.append({
                'id': item['id'],
                'report': item['report'],
                'labels': four_class_labels[idx] # 这里存的就是 [0, 2, 1, 3, ...]
            })

    return results

def group_and_analyze_labels(data_list, times=40):
    """
    1. 根据 labels (0-3组合) 分组
    2. 统计每组数量
    3. 按 Sum (真值求和) 排序
    """
    grouped_data = defaultdict(list)
    
    # --- 1. 分组 (逻辑通用，自动适配 0-3 组合) ---
    for item in data_list:
        label_key = tuple(int(x) for x in item['labels'])
        grouped_data[label_key].append(item['id'])
        
    print(f"\n====== 分组统计结果 (共 {len(grouped_data)} 种 4分类组合) ======")
    
    # --- 2. 按【样本数量】排序 (保持完全相同的组合为一组) ---
    print(f"--- [按样本数量排序] Top {times} Most Frequent Combinations ---")
    sorted_by_count = sorted(grouped_data.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (label_pattern, ids) in enumerate(sorted_by_count):
        if i < times:
            d_sum = sum(label_pattern) 
            # 这里的 Label 现在会显示如 (1, 0, 2, 3...)
            print(f"Group {i+1:04d}: Count = {len(ids):04d} | Sum = {d_sum:02d} | Label = {label_pattern}")

    # --- 3. 按【真值总和(Sum)】排序 ---
    # 这会展示“数值上”最大的组合（例如含有大量 3-Uncertain 或 2-Negative 的组合）
    print(f"\n--- [按真值总和排序] Top {times} Highest Sum (Complexity) ---")
    sorted_by_sum = sorted(grouped_data.items(), key=lambda x: sum(x[0]), reverse=True)
    
    for i, (label_pattern, ids) in enumerate(sorted_by_sum):
        if i < times:
            d_sum = sum(label_pattern)
            print(f"Rank {i+1:04d}: Sum = {d_sum:02d} | Count = {len(ids):02d} | Label = {label_pattern}")
            
    return grouped_data

# ==========================================
# 4. 配置与运行
# ==========================================
if __name__ == "__main__":
    # --- 配置区域 ---
    CKPT_DIR = "/mnt/data/liuhongyu/rg/iu_x-ray_chen/chexbert_weights"
    BERT_PATH = "/mnt/data/liuhongyu/rg/checkpoints/bert-base-uncased"
    CHECKPOINT_FILE = "/mnt/data/liuhongyu/rg/checkpoints/stanford/chexbert/chexbert.pth"
    JSON_PATH = "/mnt/data/liuhongyu/IUXRay/hergen_iuxray.json"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Initializing CheXbert on {DEVICE}...")
    try:
        if not os.path.exists(CKPT_DIR):
             print(f"Warning: {CKPT_DIR} not found. Please set correct path.")

        chexbert = CheXbert(
            ckpt_dir=CKPT_DIR,
            bert_path=BERT_PATH,
            checkpoint_path=CHECKPOINT_FILE,
            device=DEVICE
        )
        
        # --- 处理数据 ---
        train_data = process_json_with_chexbert(
            json_path=JSON_PATH, 
            model=chexbert, 
            target_split='train', 
            batch_size=32 
        )

        # --- 结果展示 ---
        if 'train_data' in locals() and train_data:
            # 这里的统计结果现在是基于 0,1,2,3 的了
            label_groups = group_and_analyze_labels(train_data, 20)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n运行出错: {e}")