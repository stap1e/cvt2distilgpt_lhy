# new_lab：On-Policy Distillation 实验线（Form A / B / C）

本目录是 `new_lab` 分支上三类蒸馏实验的唯一记录入口。所有改动见
[CHANGELOG.md](CHANGELOG.md)；每次训练后的结果分析写进
[EXPERIMENTS.md](EXPERIMENTS.md)；三个形态的设计与命令分别在
[FORM_A.md](FORM_A.md)、[FORM_B.md](FORM_B.md)、[FORM_C.md](FORM_C.md)。

## 核心设计约束

1. **教师永远不进训练进程。** 教师推理全部通过独立离线 CLI 完成
   （`tools/distillation/`），训练 GPU 只跑学生。真教师（MedGemma 等）
   在另一环境/另一张卡上运行，产出 JSONL 后用 `--teacher file:<path>`
   引入，或在该环境里直接用 `--teacher vlm:<model>`。
2. **所有 CLI 自动写 sidecar 报告**（`*.report.md`），训练自动写
   `new_lab_run_record.{jsonl,md}` 到 trial 目录。记录只含算法相关内容
   （loss 分量、指标、池统计），不含算力消耗（时间/GPU/显存已由白名单
   强制排除）。
3. **`mock` 教师只用于冒烟测试**。它产生的任何指标不得作为实验结论，
   必须在 EXPERIMENTS.md 里标注 "mock"。

## 三形态一览

| 形态 | 思路 | 离线步骤 | 训练入口 |
|---|---|---|---|
| A 教师重写 SFT | 教师重写训练报告（保持"当前可观测"），替换训练目标 | `teacher_rewrite.py` | `train_covar_distill_sft.yaml`（train_mode: ce） |
| B 在线蒸馏 GKD | 学生采样 → 教师逐 token 打分 → 稠密奖励 REINFORCE（或 best-of-K SFT） | `export_rollouts.py` → `teacher_score.py --mode tokens` | `train_covar_gkd.yaml`（train_mode: gkd） |
| C 教师偏好 DPO | 学生采样 → 教师排序构偏好对 → DPO | `export_rollouts.py` → `teacher_score.py --mode rank` → `--mode fill-ref-logps` | `train_covar_dpo_distill.yaml`（train_mode: dpo） |

## 快速开始（以 B 为例，其余见各 FORM 文档）

```bash
# 0) 冒烟：mock 教师跑通整条管线（小样本）
python -m tools.distillation.export_rollouts \
  --config config/new_lab/train_covar_gkd.yaml \
  --ckpt-path <stage1.ckpt> --split train --num-samples 2 --round 0 \
  --out /data/lhy_data/rg/results_new_lab/gkd_pool/rollouts_round0.jsonl \
  --device cuda:5 --max-items 64

python -m tools.distillation.teacher_score --mode tokens \
  --rollouts /data/lhy_data/rg/results_new_lab/gkd_pool/rollouts_round0.jsonl \
  --out /data/lhy_data/rg/results_new_lab/gkd_pool/scored_round0.jsonl \
  --config config/new_lab/train_covar_gkd.yaml --teacher mock

# 1) 训练（gkd_rollout_path 指向目录，自动取最新 round）
CUDA_VISIBLE_DEVICES=5 dlhpcstarter -t covar_gkd \
  -c config/new_lab/train_covar_gkd.yaml \
  --stages_module stages_visual_grounded_v2 --train --trial 1 \
  --warm_start_ckpt_path <stage1.ckpt>

# 2) 用新 ckpt 导出下一轮 rollouts（--round 递增），回到 0
```

## 训练后固定动作（每次训练完必须做）

1. 打开 `<exp_dir_trial>/new_lab_run_record.md`，把 epoch 指标表与
   关键曲线结论（哪些 loss 在降、monitor 何时最优）摘录进
   [EXPERIMENTS.md](EXPERIMENTS.md) 对应小节。
2. 查看对应离线步骤的 `*.report.md`（对齐覆盖率、教师-学生差距、
   偏好对胜率等），把异常值写进实验记录。
3. 跑测试并记录指标：
   `CUDA_VISIBLE_DEVICES=5 dlhpcstarter -t <task> -c <config> --stages_module stages_visual_grounded_v2 --test --trial <n>`。
4. 在 EXPERIMENTS.md 写"分析 + 结论 + 下一步"，再进入下一轮。

## 已知边界（先读再做实验）

- `teacher_score.py --mode tokens` 的跨词表对齐靠字符 span；若
  `coverage_mean < 0.9`，稠密奖励里最近邻兜底占比过高，先抽查样例。
- GKD（dense_rl）每 step 额外一次 rollout 前向；显式记录里 effective
  数据集大小变为 样本数×K，一个 epoch 的更新步数相应变化，调 LR /
  max_epochs 时要意识到这点（这是算法相关事实，不算算力消耗）。
- DPO 配置默认 `dpo_reference_free: false`；必须先跑
  `--mode fill-ref-logps`，否则 setup 阶段会直接报错（这是有意的
  防呆：避免再次落入 reference-free 冒烟模式）。
- 真教师接入两条路：`file:<path>`（推荐，环境解耦）或
  `vlm:<model>`（需要在装有新版 transformers 的独立环境运行离线 CLI）。
