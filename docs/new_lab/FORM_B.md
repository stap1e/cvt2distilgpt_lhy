# Form B：On-Policy 蒸馏（GKD，稠密教师奖励）

## 假设

学生在自身采样分布上训练、教师对采样序列逐 token 给 logprob，等价于
"以教师 logprob 为稠密奖励的 REINFORCE"（GKD / on-policy distillation 思路），
比 SCST 的序列级词法奖励信号密得多，且不会 reward hacking 手写奖励。
参考分支保留完整 COVAR 损失作锚，防止语言先验被冲掉。

由于学生（GPT2 BPE）与教师（如 SentencePiece）词表不同，全分布 KL 不可行；
可实现的是**教师对已实现 token 的 logprob**，经字符 span 对齐到学生
token（`tools/distillation/token_align.py`）。

## 目标函数（`_training_step_gkd`，objective=dense_rl）

```
r_t   = 教师对齐 logprob(y_t)                    （EOS/PAD 不监督）
adv_t = r_t - baseline   → whiten → clip(±3)
L_gkd = -Σ_t adv_t · logπ_S(y_t) / Σ_t 1
total = gkd_weight · L_gkd + gkd_reference_ce_scale · L_COVAR(参考报告)
```

baseline 两档：`sequence_mean`（默认，序列内均值）与 `greedy_score`
（教师对 greedy rollout 的平均 logprob，池里必须有 greedy 行——即导出
时不要 `--no-greedy`）。

`objective=best_of_k` 是稳定对照：每例取教师分数最高的采样 rollout 当
SFT 目标，跑完整 COVAR 损失（无 REINFORCE 方差）。

## 轮次循环（round loop）

```bash
POOL=/data/lhy_data/rg/results_new_lab/gkd_pool
CKPT=<stage1 或上一轮 ckpt>

# 1) 导出 rollouts（学生侧，任意空闲卡）
python -m tools.distillation.export_rollouts \
  --config config/new_lab/train_covar_gkd.yaml --ckpt-path $CKPT \
  --split train --num-samples 4 --round 1 \
  --out $POOL/rollouts_round1.jsonl --device cuda:5

# 2) 教师打分（mock 冒烟 / file 桥接真教师；别占训练卡）
python -m tools.distillation.teacher_score --mode tokens \
  --rollouts $POOL/rollouts_round1.jsonl \
  --out $POOL/scored_round1.jsonl \
  --config config/new_lab/train_covar_gkd.yaml --teacher mock

# 3) 训练一轮（目录里最新 round 自动生效；trial 号建议 = round 号）
CUDA_VISIBLE_DEVICES=5 dlhpcstarter -t covar_gkd \
  -c config/new_lab/train_covar_gkd.yaml \
  --stages_module stages_visual_grounded_v2 --train --trial 1 \
  --warm_start_ckpt_path $CKPT

# 4) CKPT ← 本轮最优 ckpt，round+1，回到 1
```

## 记录清单

| 文件 | 看什么 |
|---|---|
| `rollouts_roundN.report.md` | `temporal_claim_rate`（采样 rollout 的时态泄漏率）、greedy-duplicate 率、长度分布 |
| `scored_roundN.report.md` | **alignment coverage**（<0.9 先抽查对齐）、`gap_rollout_minus_reference_mean`（教师-学生差距，随轮次应收敛向 0） |
| `<exp_dir_trial>/new_lab_run_record.md` | `train_gkd_loss`（应下降）、`train_gkd_reward_token_mean`（应上升）、`train_gkd_rollout_logp_mean`、COVAR 分量是否被破坏（`train_evidence_clinical_gap` 等）、val 指标 |

## 判定与调参

- **有效信号**：`reward_token_mean` 逐轮上升 + `val_ce_f1_macro` 不降。
  只升 reward 不升指标 → 教师在奖励"更像参考的措辞"，考虑
  `gkd_advantage_baseline: greedy_score`（相对自身 greedy 改进而非模仿）。
- **奖励噪声**：`advantage_mean` 长期≈0 且 loss 不动 → 教师打分近乎恒定，
  换教师或提高采样温度让 rollout 多样性上来。
- **语言崩坏**：`rollout_logp_mean` 暴跌 / 生成重复 → 降
  `gkd_weight`（0.5→0.2）或升 `gkd_reference_ce_scale`。
- **方差过大**：换 `best_of_k` 对照跑一轮，若稳定优于 dense_rl，
  优先保守路线。

## 风险

- 教师必须**看图**打分（mock 看不到图，只能测管线）。文本教师会把
  幻觉当流畅奖励。
- off-policy 陈旧性：一个 round 内 ckpt 会漂移。已导出
  `student_logp` 留作后续重要性加权接口；当前靠"多轮次+小 epoch"缓解。
- epoch 长度 = 样本数 × K，round 1 的 mbatch 步数约是 CE 训练的 K+1 倍，
  `max_epochs: 4` 的实际更新量需在记录里换算说明。
