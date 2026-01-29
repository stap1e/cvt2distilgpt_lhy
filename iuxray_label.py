import json
import os
import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict, Counter
from transformers import BertConfig, BertModel, BertTokenizer
from tqdm import tqdm 

def inject_labels(dataset_split):
    if dataset_split is None: return
    for sample in dataset_split:
        # [关键] 必须找到能与 disease_map 的 key (如 "CXR1_1_IM-0001") 匹配的 ID
        # 假设 sample['id'] 存在且就是那个 key
        # 如果 key 隐藏在 image_path 里，需要做类似 os.path.basename 的处理
        key = sample.get('id') 
        
        # 如果你的 key 是文件名，请取消下面这行的注释并修改逻辑：
        # key = os.path.splitext(os.path.basename(sample['image_path'][0]))[0]

        if key and key in disease_map:
            sample['disease_labels'] = disease_map[key]
        else:
            # 如果没找到对应标签，填充全0或忽略 (这里填14个0作为默认)
            # 这里的 0 代表 "Negative" 还是 "Unmentioned" 取决于你的编码逻辑
            sample['disease_labels'] = [0] * 14 
            # print(f"Warning: No labels found for {key}")

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
# 2. 标签转换逻辑 (保持不变)
# ==========================================
def convert_chexbert_output_to_binary(preds, policy='u_zeros'):
    binary_preds = torch.zeros_like(preds, dtype=torch.float32)
    positive_mask = (preds[:, :13] == 1)
    uncertain_mask = (preds[:, :13] == 3)
    binary_preds[:, :13][positive_mask] = 1.0
    if policy == 'u_ones': binary_preds[:, :13][uncertain_mask] = 1.0
    else: binary_preds[:, :13][uncertain_mask] = 0.0
    no_disease = (binary_preds[:, :13].sum(dim=1) == 0)
    binary_preds[no_disease, 13] = 1.0
    return binary_preds

# ==========================================
# 3. 数据处理主流程
# ==========================================
# def process_json_with_chexbert(json_path, model, target_split='train', batch_size=32):
#     if not os.path.exists(json_path):
#         raise FileNotFoundError(f"JSON file not found at {json_path}")
        
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     # 智能解析结构
#     if isinstance(data, list): data_src = data
#     elif isinstance(data, dict):
#         if 'images' in data: data_src = data['images']
#         elif target_split in data: data_src = data[target_split]
#         else: data_src = list(data.values())
#     else: raise ValueError("Unknown JSON format")

#     # 筛选 split
#     valid_samples = []
#     for item in data_src:
#         if not isinstance(item, dict): continue
#         curr_split = item.get('split', target_split)
#         if curr_split == target_split:
#             report = item.get('report', "")
#             if report:
#                 valid_samples.append({
#                     'id': item.get('id', 'unknown'),
#                     'report': report
#                 })

#     print(f"Split '{target_split}' 筛选出 {len(valid_samples)} 条有效样本。开始 CheXbert 推理...")

#     results = []
#     for i in tqdm(range(0, len(valid_samples), batch_size), desc="Inferencing"):
#         batch_items = valid_samples[i : i + batch_size]
#         batch_reports = [item['report'] for item in batch_items]
        
#         raw_preds = model(batch_reports)
#         binary_labels = convert_chexbert_output_to_binary(raw_preds, policy='u_zeros')
        
#         for idx, item in enumerate(batch_items):
#             results.append({
#                 'id': item['id'],
#                 'report': item['report'],
#                 'labels': binary_labels[idx].cpu().tolist()
#             })
#     return results

def process_json_with_chexbert(json_path, model, target_split='train', batch_size=32):
    # ... (前面的读取 JSON 代码保持不变) ...
    
    # [这里省略了文件读取部分，和之前一样]
    if not os.path.exists(json_path): raise FileNotFoundError(f"JSON file not found at {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
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

    print(f"Split '{target_split}' 筛选出 {len(valid_samples)} 条有效样本。开始提取 4分类 标签...")

    results = []
    for i in tqdm(range(0, len(valid_samples), batch_size), desc="Inferencing"):
        batch_items = valid_samples[i : i + batch_size]
        batch_reports = [item['report'] for item in batch_items]
        
        # 1. 模型推理 (得到原始的 0-3 整数)
        # raw_preds shape: [Batch, 14]
        # 里面的值已经是 0, 1, 2, 3 了
        raw_preds = model(batch_reports) 
        
        # 2. 【核心修改】直接保存 raw_preds，不要调用 convert_chexbert_output_to_binary
        #    这样你就得到了 Unmentioned(0), Positive(1), Negative(2), Unclear(3)
        four_class_labels = raw_preds.cpu().tolist()
        
        # 3. 存回结果
        for idx, item in enumerate(batch_items):
            results.append({
                'id': item['id'],
                'report': item['report'],
                'labels': four_class_labels[idx] # 这里存的就是 [0, 1, 0, 3, ...]
            })

    return results


def group_and_analyze_labels(data_list, times=40):
    grouped_data = defaultdict(list)
    for item in data_list:
        label_key = tuple(int(x) for x in item['labels'])
        grouped_data[label_key].append(item['id'])
        
    print(f"\n====== 分组统计结果 (共 {len(grouped_data)} 种组合) ======")
    
    print(f"--- [按样本数量排序] Top {times} Most Frequent ---")
    sorted_by_count = sorted(grouped_data.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (label_pattern, ids) in enumerate(sorted_by_count):
        if i < times:
            d_sum = sum(label_pattern) 
            print(f"Group {i+1:04d}: Count = {len(ids):04d} | Sum = {d_sum:02d} | Label = {label_pattern}")

    print(f"\n--- [按疾病总数排序] Top {times} Most Complex (Sum) ---")
    sorted_by_sum = sorted(grouped_data.items(), key=lambda x: sum(x[0]), reverse=True)
    for i, (label_pattern, ids) in enumerate(sorted_by_sum):
        if i < times:
            d_sum = sum(label_pattern)
            print(f"Rank {i+1:04d}: Sum = {d_sum:02d} | Count = {len(ids):02d} | Label = {label_pattern}")    
    return grouped_data

# ==========================================
# 4. 【新增】保存标签功能
# ==========================================
def save_labels_to_json(data_list, original_json_path, filename="multi_hot_label.json"):
    """
    将生成的标签保存到原始 JSON 同级目录下。
    保存格式为字典：{ "image_id": [0, 1, 0...] }，以便快速索引且节省空间。
    """
    # 1. 确定保存路径
    dir_path = os.path.dirname(original_json_path)
    save_path = os.path.join(dir_path, filename)
    
    # 2. 格式化数据：去掉 report，只保留 id 和 labels
    # 使用字典格式 {id: labels} 是最推荐的做法，因为 Dataset 读取时按 ID 查找最快
    output_data = {item['id']: item['labels'] for item in data_list}
    
    # 如果你必须保存为列表格式 [{'id': 'xxx', 'labels': []}]，请取消下面这行的注释并注释掉上面一行
    # output_data = [{'id': item['id'], 'labels': item['labels']} for item in data_list]

    print(f"\n正在保存标签到文件: {save_path} ...")
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        print(f"✅ 保存成功！共保存 {len(output_data)} 条数据的标签。")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

# ==========================================
# 5. 配置与运行
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
        
        # --- 处理数据 (Train Split) ---
        # 注意：这里目前只处理了 'train' split。如果想处理所有数据，
        # 需要修改 process_json_with_chexbert 的逻辑或在这里循环调用不同 split
        train_data = process_json_with_chexbert(
            json_path=JSON_PATH, 
            model=chexbert, 
            target_split='train', 
            batch_size=32 
        )

        # --- 统计展示 ---
        if 'train_data' in locals() and train_data:
            label_groups = group_and_analyze_labels(train_data, 20)
            
            # --- 【关键步骤】保存结果 ---
            # 这会将 ['id'] 和 ['labels'] 保存到 multi_hot_label.json
            save_labels_to_json(train_data, JSON_PATH)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n运行出错: {e}")