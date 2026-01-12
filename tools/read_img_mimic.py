import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def main():
    # 数据集根目录
    img_path = '/data/lhy_data/MIMIC-CXR/files'
    
    # 支持的图片格式 (MIMIC-CXR JPG版通常是 .jpg, DICOM版是 .dcm)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.dcm')
    
    total_images = 0
    total_patient_folders = 0  # 统计 p10000032 这种病人文件夹
    total_study_folders = 0    # 统计 s50414267 这种检查文件夹 (如果存在)

    print(f"开始扫描路径: {img_path} ...")

    if not os.path.exists(img_path):
        print(f"错误: 路径不存在 {img_path}")
        return

    # os.walk 会递归遍历目录下所有的文件夹和文件
    # root: 当前正在遍历的文件夹路径
    # dirs: 当前文件夹下的子文件夹名列表
    # files: 当前文件夹下的文件名列表
    for root, dirs, files in os.walk(img_path):
        
        # 1. 统计图片数量
        for file in files:
            if file.lower().endswith(valid_extensions):
                total_images += 1
        
        # 2. 统计文件夹数量 (针对 MIMIC 结构进行特定逻辑统计)
        # 获取当前文件夹的名字
        current_folder_name = os.path.basename(root)
        
        # MIMIC-CXR 结构通常是: p10 (组) -> p10000032 (病人) -> s50414267 (检查)
        
        # 判断是否是病人文件夹 (以 'p' 开头且长度大于3，排除 p10, p11 这种组文件夹)
        # p10 长度是3，p10000032 长度是9
        if current_folder_name.startswith('p') and len(current_folder_name) > 3 and current_folder_name[1:].isdigit():
            total_patient_folders += 1
            
        # 判断是否是检查(Study)文件夹 (以 's' 开头)
        elif current_folder_name.startswith('s') and current_folder_name[1:].isdigit():
            total_study_folders += 1

    print("-" * 30)
    print("统计完成！")
    print(f"病人子文件夹数 (Patient folders, pXXXXXXX): {total_patient_folders}")
    if total_study_folders > 0:
        print(f"检查子文件夹数 (Study folders, sXXXXXXX):   {total_study_folders}")
    print(f"总图片数量 (Total Images):                 {total_images}")
    print("-" * 30)

def main2():
    img_path = '/data/lhy_data/IUXRay/image/CXR97_IM-2460/0.png'
    # 1. 读取为 PIL Image 对象 (RGB模式)
    # .convert('RGB') 是为了防止图片是 4通道(PNG透明) 或 单通道(灰度)，强制转为3通道
    img_pil = Image.open(img_path).convert('RGB')

    # ================= 转为 NumPy =================
    # 形状: (Height, Width, Channel) | 范围: 0-255 | 类型: uint8
    img_numpy = np.array(img_pil)

    # ================= 转为 Tensor =================
    # 形状: (Channel, Height, Width) | 范围: 0.0-1.0 | 类型: float32
    transform = transforms.ToTensor()
    img_tensor = transform(img_pil)

    print(f"Numpy Shape: {img_numpy.shape}, Type: {img_numpy.dtype}, Max: {img_numpy.max()}")
    print(f"Tensor Shape: {img_tensor.shape}, Type: {img_tensor.dtype}, Max: {img_tensor.max()}")

if __name__ == "__main__":
    main2()