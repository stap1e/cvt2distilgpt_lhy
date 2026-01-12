import os
import sys
from argparse import Namespace

# 假设 stages.py 在当前目录下
from stages import stages 

def main():
    # --- 0. 硬件环境设置 ---
    # 这里设置显卡编号，必须与 args.cuda_visible_devices 保持一致
    # 你提供的 log 显示是 '2'，所以这里设为 '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    # 解决部分库冲突
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # --- 1. 模拟完整参数 (Namespace) ---
    # 基于你提供的最终运行状态整理
    args = Namespace(
        # ===========================
        # 1. 任务与模式 (Task & Mode)
        # ===========================
        task="iu_x_ray_chen",
        config="config/train_iu_x_ray_chen_cvt2distilgpt2.yaml",
        stages_module="stages",
        
        # 运行模式开关
        train=True, 
        test=False,
        submit=False,      # 必须为 False，否则尝试提交集群
        multirun=False,    # 单次运行
        one_epoch_only=False, # 设为 True 可快速测试流程
        fast_dev_run=False,   # PL 的快速调试模式
       
        # ===========================
        # 2. 硬件与并行 (Hardware)
        # ===========================
        seed=9223,
        cuda_visible_devices="2", # 必须与 os.environ 一致
        accelerator="gpu",
        devices=1,
        gpus=1,
        num_nodes=1,
        num_workers=4,       # 调试时如果报错，可改为 0 (主线程加载)
        strategy="auto",
        precision=16,        # 半精度训练

        # ===========================
        # 3. 路径配置 (Paths)
        # ===========================
        # 根目录与数据
        dataset_dir="/data/lhy_data/Cvt2distillgpt2",
        ckpt_zoo_dir="/data/lhy_data/Cvt2distillgpt2/checkpoints",
        
        # 实验日志输出
        exp_dir="/data/lhy_data/Cvt2distillgpt2/results_iuxray",
        exp_dir_trial="/data/lhy_data/Cvt2distillgpt2/results_iuxray/trial_0",
        config_name="train_iu_x_ray_chen_cvt2distilgpt2",
        trial=0,

        # 热启动与恢复 (调试通常设为 None/False)
        warm_start_modules=False,
        warm_start_ckpt_path=None,
        # warm_start_modules = True,  
        # warm_start_ckpt_path='/data/lhy_data/Cvt2distillgpt2/results_iuxray/iu_x_ray_chen/train_iu_x_ray_chen_cvt2distilgpt2/trial_0/epoch=18-step=3287-val_chen_cider=0.453558.ckpt',
        warm_start_exp_dir=None,
        resume_last=False,
        resume_epoch=None,
        resume_ckpt_path=None,
        other_exp_dir=None,

        

        # ===========================
        # 4. 模型超参数 (Model Hyperparams)
        # ===========================
        module="cvt2distilgpt2_iu_x_ray_chen",
        definition="CvT2DistilGPT2IUXRayChen",
        
        encoder_lr=5e-05,
        decoder_lr=0.0005,
        mbatch_size=12,
        decoder_max_len=128,
        num_test_beams=4,
        max_epochs=100,
        every_n_epochs=1,
        deterministic=False,

        # ===========================
        # 5. 监控与回调 (Monitor & Callbacks)
        # ===========================
        monitor="val_chen_cider",
        monitor_mode="max",
        enable_progress_bar=True,
        weights_summary="full",
        early_stopping=True,
        patience=20,
        min_delta=0.0001,
    )

    # --- 2. 手动触发函数 ---
    print(f">>> [Debug Start] Task: {args.task}")
    print(f">>> GPU Device: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f">>> Config: {args.config}")
    
    try:
        stages(args)
    except Exception as e:
        print(f"\n!!! 调试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main()