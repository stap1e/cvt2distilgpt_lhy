import os

def count_dataset_stats(dataset_path):
    # 初始化统计数据的字典
    # 键 0-6 对应包含 0 到 6 张图片的文件夹数量
    count_stats = {i: 0 for i in range(7)}
    # 作为一个保险，如果有文件夹超过6张图，记在这里
    count_stats['>6'] = 0
    
    total_subdirs = 0
    total_images = 0
    
    # 定义需要统计的图片后缀 (不区分大小写)
    valid_extensions = ('.jpg', '.jpeg', '.png')

    # 检查路径是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 路径 '{dataset_path}' 不存在，请检查路径是否正确。")
        return

    print(f"正在扫描路径: {dataset_path} ...")

    # 遍历主目录下的所有项目
    try:
        items = os.listdir(dataset_path)
    except Exception as e:
        print(f"❌ 无法读取目录: {e}")
        return

    for item_name in items:
        item_path = os.path.join(dataset_path, item_name)

        # 我们只关心“子文件夹”
        if os.path.isdir(item_path):
            total_subdirs += 1
            
            # 计算当前子文件夹内的图片数量
            current_img_count = 0
            # 遍历子文件夹内的文件
            for file_name in os.listdir(item_path):
                # 检查后缀名 (转换为小写比较，避免 .PNG 和 .png 的问题)
                if file_name.lower().endswith(valid_extensions):
                    current_img_count += 1
            
            # 更新总图片数
            total_images += current_img_count
            
            # 更新分布统计
            if current_img_count <= 6:
                count_stats[current_img_count] += 1
            else:
                count_stats['>6'] += 1

    # --- 打印结果 ---
    print("\n" + "="*40)
    print("📊 数据集统计结果")
    print("="*40)
    print(f"子文件夹总数: {total_subdirs}")
    print(f"图片文件总数: {total_images}")
    print("-" * 40)
    for i in range(7):
        print(f"  - 包含 {i} 张图片的文件夹数量: {count_stats[i]}")
    
    if count_stats['>6'] > 0:
        print(f"  - 包含 >6 张图片的文件夹数量: {count_stats['>6']}")
    print("="*40)

if __name__ == "__main__":
    # 这里设置你的路径
    # 注意：如果你的脚本不在 dataset 的上一级目录运行，请修改为绝对路径
    path = "/data/lhy_data/IUXRay/image" 
    
    count_dataset_stats(path)