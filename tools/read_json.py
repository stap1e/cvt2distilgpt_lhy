import json
import re
from collections import Counter
import pandas as pd

# ================= 配置区域 =================
# 替换为你的 JSON 文件路径
JSON_FILE_PATH = '/data/lhy_data/IUXRay/hergen_iuxray.json' 

# 定义你需要重点关注的医学关键词 (根据 CheXpert 或常见病理)
TARGET_KEYWORDS = [
    'pneumothorax', 'atelectasis', 'effusion', 'opacity', 
    'consolidation', 'edema', 'cardiomegaly', 'pneumonia', 
    'fracture', 'lesion', 'mass', 'nodule', 'granuloma',
    'hernia', 'device', 'catheter', 'tube'
]

# 定义停用词 (这些词频率极高但没太多实际含义)
STOP_WORDS = {
    'the', 'and', 'is', 'of', 'in', 'to', 'are', 'a', 'with', 'for', 
    'no', 'or', 'on', 'at', 'be', 'this', 'that', 'as', 'by', 'it', 'there'
}
# ===========================================

def load_data(file_path, target_keys=None):
    """
    读取 JSON 数据。
    
    Args:
        file_path (str): JSON 文件路径
        target_keys (str or list): 指定要读取的键名。
                                   - 如果是字符串：例如 'train'
                                   - 如果是列表：例如 ['train', 'val']
                                   - 如果是 None：则读取字典中所有的列表（保持原逻辑）
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 情况1：如果文件根节点本身就是列表（不是字典），直接返回，无法按 Key 筛选
        if isinstance(data, list):
            return data

        # 情况2：文件是字典，开始筛选
        if isinstance(data, dict):
            all_items = []
            
            # --- 核心修改部分开始 ---
            
            # A. 如果指定了关键字
            if target_keys is not None:
                # 将单个字符串转为列表，方便统一循环处理
                if isinstance(target_keys, str):
                    keys_to_load = [target_keys]
                else:
                    keys_to_load = target_keys
                
                print(f"当前筛选模式：只读取 {keys_to_load}")
                
                for key in keys_to_load:
                    if key in data:
                        if isinstance(data[key], list):
                            all_items.extend(data[key])
                            print(f" -> 成功加载 '{key}'，包含 {len(data[key])} 条数据")
                        else:
                            print(f" -> 跳过 '{key}'：内容不是列表")
                    else:
                        print(f" -> 警告：JSON 文件中找不到键 '{key}'")

            # B. 如果没指定关键字 (None)，读取所有是列表的值 (原逻辑)
            else:
                print("模式：读取所有包含列表的键")
                for key in data:
                    if isinstance(data[key], list):
                        all_items.extend(data[key])
            
            # --- 核心修改部分结束 ---
            
            return all_items
            
        return [] # 既不是 list 也不是 dict
        
    except Exception as e:
        print(f"读取文件失败: {e}")
        return []

def preprocess_text(text):
    """文本清洗：转小写，去标点"""
    if not text:
        return ""
    text = text.lower()
    # 将所有非字母数字字符替换为空格
    text = re.sub(r'[^\w\s]', ' ', text)
    # 去除多余空格
    return text

def get_bigrams(words):
    """生成双词短语 (Bigrams)，例如 'pleural effusion'"""
    return [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]

def main():
    # 1. 读取数据
    print(f"正在读取文件: {JSON_FILE_PATH} ...")
    data = load_data(JSON_FILE_PATH, target_keys=None)
    print(f"共加载 {len(data)} 条报告数据。\n")

    # 2. 提取并清洗所有文本
    all_words = []
    
    for item in data:
        # 确保 key 是 report (根据你的数据结构)
        report_text = item.get('report', '') 
        clean_t = preprocess_text(report_text)
        # 分词
        words = clean_t.split()
        all_words.extend(words)

    total_word_count = len(all_words)
    print(f"预处理完成，语料库共包含 {total_word_count} 个单词。\n")

    # 3. 统计词频 (Counter)
    counter = Counter(all_words)

    # --- 分析 A: 除去停用词后的通用高频词 ---
    print("【Top 20 高频单词 (已去除停用词)】")
    print("-" * 40)
    # 过滤停用词
    filtered_counter = {k: v for k, v in counter.items() if k not in STOP_WORDS}
    # 重新转回 Counter 对象以便排序
    final_counter = Counter(filtered_counter)
    
    for word, count in final_counter.most_common(20):
        # 计算占比
        ratio = (count / total_word_count) * 100
        print(f"{word:<20} | 次数: {count:<6} | 占比: {ratio:.2f}%")
    print("\n")

    # --- 分析 B: 指定医学关键词统计 ---
    print("【指定医学关键词统计】")
    print("-" * 40)
    results = []
    for keyword in TARGET_KEYWORDS:
        count = counter.get(keyword, 0)
        results.append({'关键词': keyword, '出现次数': count})
    
    # 使用 Pandas 展示更美观
    df_med = pd.DataFrame(results).sort_values(by='出现次数', ascending=False)
    print(df_med)
    print("\n")

    # --- 分析 C: 双词短语 (词组) 统计 ---
    # 这对于医学报告非常重要，因为 "pleural effusion" 比单独的 "pleural" 更能说明问题
    print("【Top 10 常见双词短语 (Bigrams)】")
    print("-" * 40)
    all_bigrams = get_bigrams([w for w in all_words if w not in STOP_WORDS])
    bigram_counter = Counter(all_bigrams)
    
    for bigram, count in bigram_counter.most_common(10):
        print(f"{bigram:<25} | 次数: {count}")

if __name__ == "__main__":
    main()  