import json
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import BertConfig, BertModel, BertTokenizer
from tqdm import tqdm 

# ==========================================
# 1. CheXbert 模型定义 (保持不变)
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
            state_dict = torch.load(full_ckpt_path, map_location=device)['model_state_dict']
            new_state_dict = OrderedDict()
            new_state_dict["bert.embeddings.position_ids"] = torch.arange(config.max_position_embeddings).expand((1, -1))
            for key, value in state_dict.items():
                if 'bert' in key: new_key = key.replace('module.bert.', 'bert.')
                elif 'linear_heads' in key: new_key = key.replace('module.linear_heads.', 'linear_heads.')
                else: new_key = key
                new_state_dict[new_key] = value
            self.load_state_dict(new_state_dict, strict=False)
        self.eval().to(device)

    def forward(self, reports):
        processed_reports = [r.strip().replace("\n", " ").replace("\s+", " ") if isinstance(r, str) else "" for r in reports]
        if not processed_reports: return None
        with torch.no_grad():
            tokenized = self.tokenizer(processed_reports, padding='longest', truncation=True, max_length=512, return_tensors="pt")
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            last_hidden_state = self.bert(**tokenized)[0]
            cls = self.dropout(last_hidden_state[:, 0, :])
            predictions = [self.linear_heads[i](cls).argmax(dim=1) for i in range(14)]
        return torch.stack(predictions, dim=1)

# ==========================================
# 2. 核心处理逻辑 (修改为只保存 ID: Labels)
# ==========================================
def process_train_only_simplified(input_json_path, output_json_path, model, batch_size=64):
    print(f"读取输入数据: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取 train 数据
    if isinstance(data, dict):
        train_items = data.get('train', [])
    else:
        print("警告：输入格式不是字典，默认全部视为训练数据")
        train_items = data

    if not train_items:
        print("未找到训练数据！")
        return

    # 结果字典： { "image_id_string": [0, 1, 2, ...] }
    id_label_map = {}

    print(f"开始处理 Train 集 (共 {len(train_items)} 条)...")
    
    for i in tqdm(range(0, len(train_items), batch_size), desc="Inferencing"):
        batch = train_items[i : i + batch_size]
        reports = [item.get('report', "") for item in batch]
        
        try:
            preds = model(reports) # [Batch, 14]
            labels = preds.cpu().tolist()
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            labels = [[0]*14] * len(batch)

        # 将结果存入字典，Key=ID, Value=Labels
        for idx, item in enumerate(batch):
            image_id = item.get('id')
            if image_id:
                id_label_map[image_id] = labels[idx]

    print(f"正在保存精简版 JSON 到: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(id_label_map, f, indent=4)
    print(f"完成！共保存 {len(id_label_map)} 条数据的标签。")

# ==========================================
# 3. 运行配置
# ==========================================
if __name__ == "__main__":
    # 输入与输出路径
    INPUT_JSON_PATH = "/mnt/data/liuhongyu/MIMIC-CXR/mimic_annotation.json" 
    OUTPUT_JSON_PATH = "/mnt/data/liuhongyu/rg/mimic_cxr_chen/mimic_cxr_chexbert_labeled.json"

    # 模型配置
    CKPT_DIR = "/mnt/data/liuhongyu/rg/iu_x-ray_chen/chexbert_weights"
    BERT_PATH = "/mnt/data/liuhongyu/rg/checkpoints/bert-base-uncased"
    CHECKPOINT_FILE = "/mnt/data/liuhongyu/rg/checkpoints/stanford/chexbert/chexbert.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(INPUT_JSON_PATH):
        print(f"Initializing CheXbert on {DEVICE}...")
        chexbert = CheXbert(CKPT_DIR, BERT_PATH, CHECKPOINT_FILE, DEVICE)
        process_train_only_simplified(INPUT_JSON_PATH, OUTPUT_JSON_PATH, chexbert, batch_size=128)
    else:
        print(f"找不到输入文件: {INPUT_JSON_PATH}")