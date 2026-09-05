# Form C：教师偏好排序 + DPO（on-policy 偏好蒸馏）

## 假设

学生采 K 条 rollout，教师（最好看图）做排序，构 chosen/rejected 偏好对
喂 DPO——既绕开跨词表对齐问题，又把 DPO 数据从"参考 vs 自己"升级为
"学生自己的好坏对比"（chosen 不再恒为 reference，能学到
"两者都有缺陷时哪个更好"）。ref-logp 由 fill-ref-logps 用 warm-start
ckpt 离线补齐，彻底告别 reference-free 冒烟模式。

## 管线

```
export_rollouts.py（greedy + K samples）
        │
teacher_score.py --mode rank     → pairs_roundN.jsonl + rank report
        │                          (chosen: winner/best_sample/reference
        │                           rejected: worst_sample/greedy)
teacher_score.py --mode fill-ref-logps → pairs_roundN_ref.jsonl
        │
train_covar_dpo_distill.yaml（train_mode: dpo, dpo_reference_free: false）
```

chosen 策略说明：
- `winner`（默认）：参考与最优 rollout 中教师更喜欢者 → 混合模仿与超越；
- `best_sample`：纯 on-policy chosen；
- `reference`：退回旧设定，作为消融对照。

## 命令

```bash
DPO=/data/lhy_data/rg/results_new_lab/dpo
CKPT=<stage1 ckpt>

python -m tools.distillation.export_rollouts \
  --config config/new_lab/train_covar_dpo_distill.yaml --ckpt-path $CKPT \
  --split train --num-samples 4 --round 1 \
  --out $DPO/rollouts_round1.jsonl --device cuda:5

python -m tools.distillation.teacher_score --mode rank \
  --rollouts $DPO/rollouts_round1.jsonl \
  --out $DPO/pairs_round1.jsonl \
  --config config/new_lab/train_covar_dpo_distill.yaml \
  --teacher mock --chosen-policy winner --rejected-policy worst_sample

python -m tools.distillation.teacher_score --mode fill-ref-logps \
  --pairs $DPO/pairs_round1.jsonl \
  --out $DPO/pairs_round1_ref.jsonl \
  --config config/new_lab/train_covar_dpo_distill.yaml \
  --ckpt-path $CKPT --device cuda:5

CUDA_VISIBLE_DEVICES=5 dlhpcstarter -t covar_dpo_distill \
  -c config/new_lab/train_covar_dpo_distill.yaml \
  --stages_module stages_visual_grounded_v2 --train --trial 1 \
  --warm_start_ckpt_path $CKPT
```

纯基线对照：`config/new_lab/train_baseline_dpo_distill.yaml`（同对文件，
`--stages_module stages`，task `baseline_dpo_distill`）。

## 记录清单

| 文件 | 看什么 |
|---|---|
| `rollouts_roundN.report.md` | 同 Form B |
| `pairs_roundN.report.md` | **best_sample_beats_reference_rate**（>0.5 说明 on-policy 数据有超越模仿的信号）、skipped_gap 比例、chosen-rejected 分差分布 |
| `pairs_roundN_ref.report.md` | ref_logp chosen/rejected 均值（差异过大说明对偏斜） |
| `<exp_dir_trial>/new_lab_run_record.md` | `train_dpo_loss` 下降、`dpo_margin` 上升、`train_dpo_reward_chosen - rejected`（教师视角的对分离度，应增大）、`dpo_pair_coverage`、`train_ce_loss`（防语言崩坏锚） |

## 判定与调参

- margin 升但 test CE F1 不动 → β 太小或对太易（分差大但无信息量），
  提高 `--min-score-gap` 保留难对，或降 K 提质量。
- `reward_chosen ≈ reward_rejected` → 教师分辨力不足（mock 必然如此），
  换真教师后再下结论。
- 语言崩坏（CE 快速上涨、生成重复）→ 升 `dpo_ce_weight`（0.1→0.3）
  或降 LR；COVAR 版还可观察 `val_temporal_claim_rate` 是否守住。
- `dpo_pair_coverage` 低 → 排名阶段 skip 太多，看 skipped_gap。

## 风险

- 教师排序若不看图，偏好会偏向"像参考"而非"符合图像"——结论会系统性
  偏向模仿。真教师接入前，mock 结论只用于管线验证。
- fill-ref-logps 用 train 图像逐对前向；pairs 很大时它在 GPU 上是一次性
  离线成本，用空闲卡跑。
