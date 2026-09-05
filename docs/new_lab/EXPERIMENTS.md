# 实验记录（EXPERIMENTS.md）

每次训练/评测后**必须**追加一节。模板在最下面，先写规范：

## 填写规范

1. **一节 = 一次可复现运行**：标题含日期、形态、round/trial、教师类型
   （mock 结果必须显式标注，不得与真教师结论混排）。
2. **数字来源必须可追溯**：写明来自哪个文件
   （`new_lab_run_record.md` 第几 epoch、哪个 `*.report.md`、哪个
   trial 的 test 输出）。本文件只放摘录，原文不删。
3. **分析三段式**：现象（指标怎么动）→ 归因（哪个组件导致，引用 loss
   分量）→ 决策（下一步改哪个超参/组件，预期什么变化）。
4. **对照优先**：任何结论都要有对照（同预算的 stage-1 CE 复训、
   best_of_k vs dense_rl、winner vs reference）。没有对照的数字只能
   记为观察。
5. 只记算法相关内容；不要写训练耗时/卡型等算力信息。

## 重点关注指标速查

| 形态 | 训练期信号 | 判定指标 | 守门指标（不得劣化） |
|---|---|---|---|
| A | train_ce_loss、concept_state_accuracy | test CE F1 macro、CIDEr | temporal_claim_rate、evidence_clinical_gap |
| B | gkd_loss↓、gkd_reward_token_mean↑、gkd_rollout_logp_mean | test CE F1 macro、gap_rollout_minus_reference 收敛 | rollout_logp_mean（不暴跌）、val_cider |
| C | dpo_loss↓、dpo_margin↑、reward 分离度↑ | test CE F1 macro | train_ce_loss、temporal_claim_rate、val_cider |

---

## 实验记录

<!-- 模板（复制后填写）：

### 2026-XX-XX · Form B round 1 · trial 1 · teacher=?? · seed 9223

**设置**
- 配置：config/new_lab/train_covar_gkd.yaml（改动项：…）
- warm-start ckpt：<路径>
- 数据：scored_round1.jsonl（N 例 / M 条，K=4，来源 rollouts_round1.jsonl）
- 关键超参：objective=dense_rl, weight=1.0, baseline=sequence_mean, …

**离线阶段记录**（摘自 *.report.md）
- alignment coverage：…
- gap_rollout_minus_reference_mean：…
- best_sample_beats_greedy_rate（如有）：…

**训练期记录**（摘自 new_lab_run_record.md）
| epoch | train_ce | gkd_loss | reward_token_mean | rollout_logp_mean | val_ce_f1_macro | val_cider |
|---|---|---|---|---|---|---|
| 0 | | | | | | |
| best | | | | | | |

**测试结果**（trial_X test 输出）
| 指标 | 本次 | stage-1 对照 | Δ |
|---|---|---|---|
| test CE F1 macro | | | |
| test CIDEr | | | |
| temporal_claim_rate | | | |

**分析**
- 现象：…
- 归因：…
- 决策：…（下一轮改什么、预期信号）

-->

（尚无正式实验记录——首条记录从第一次非 mock 运行开始。）
