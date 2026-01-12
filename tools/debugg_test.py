import os
import sys
from argparse import Namespace

# 假设 stages.py 在当前目录下
from stages import stages 

def main():
    # --- 0. 硬件环境设置 ---
    # 根据你的命令行 CUDA_VISIBLE_DEVICES=1
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    # 解决部分 Tokenizers 库并行冲突
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # --- 1. 模拟完整参数 (Namespace) ---
    args = Namespace(
        # ===========================
        # 1. 任务与模式 (Task & Mode)
        # ===========================
        task="mimic_cxr_chen",
        # 注意：这里改为 test 的配置文件
        config="config/test_mimic_cxr_chen_cvt2distilgpt2.yaml", 
        stages_module="stages",
        
        # 关键开关：测试模式
        train=False, 
        test=True,            # <--- 开启测试
        submit=False,
        multirun=False,
        one_epoch_only=False,
        fast_dev_run=False,

        # ===========================
        # 2. 关键路径 (Paths & Checkpoint)
        # ===========================

        # 根目录与数据
        dataset_dir="/data/lhy_data/Cvt2distillgpt2",
        ckpt_zoo_dir="/data/lhy_data/Cvt2distillgpt2/checkpoints",
        
        # 实验输出目录
        exp_dir="/data/lhy_data/Cvt2distillgpt2/results_mimic_cxr",
        # 注意：测试时通常不需要具体的 trial 目录，但为了保持一致性可以指向原实验目录
        exp_dir_trial="/data/lhy_data/Cvt2distillgpt2/results_mimic_cxr/mimic_cxr_chen/train_mimic_cxr_chen_cvt2distilgpt2/trial_0",
        config_name="train_mimic_cxr_chen_cvt2distilgpt2",
        trial=0,
        # 【核心】指定要测试的权重文件路径
        test_ckpt_path="/data/lhy_data/Cvt2distillgpt2/results_mimic_cxr/mimic_cxr_chen/train_mimic_cxr_chen_cvt2distilgpt2/trial_0/epoch=14-step=1920-val_chen_cider=0.535580.ckpt",

        # 其他热启动参数（测试模式下通常设为 None，因为主要依赖 test_ckpt_path）
        warm_start_modules=False,
        warm_start_ckpt_path=None,
        warm_start_exp_dir=None,
        resume_last=False,
        resume_epoch=None,
        resume_ckpt_path=None,
        other_exp_dir=None,
        test_epoch=None,

        # ===========================
        # 3. 硬件与并行 (Hardware)
        # ===========================
        seed=9223,
        cuda_visible_devices="2", # 必须与 os.environ 保持一致
        accelerator="gpu",
        devices=1,
        gpus=1,
        num_nodes=1,
        num_workers=4,            # YAML 中指定为 5
        # YAML 中指定了特殊的 strategy
        strategy="auto", 
        precision=16,             # 保持与训练一致

        # ===========================
        # 4. 模型超参数 (Model Hyperparams)
        # ===========================
        module="cvt2distilgpt2_mimic_cxr_chen",
        definition="CvT2DistilGPT2MIMICXRChen",
        
        encoder_lr=5e-05,
        decoder_lr=0.0005,
        mbatch_size=4,           # YAML 中指定为 4
        decoder_max_len=128,
        num_test_beams=4,
        
        # 以下参数在测试阶段通常不影响，但为了防止报错保留
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
        patience=10,
        min_delta=1e-4,
    )

    # --- 2. 手动触发函数 ---
    print(f">>> [Debug Test Start] Task: {args.task}")
    print(f">>> Loading Checkpoint: {args.test_ckpt_path}")
    print(f">>> GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    try:
        stages(args)
    except Exception as e:
        print(f"\n!!! 测试调试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main()