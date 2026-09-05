# new_lab 分支修改记录（CHANGELOG）

本文件记录 `new_lab` 分支相对 `baseline` 分支（commit `246f530`, COVAR v2.2）
的**全部**代码改动与理由。按"新增文件 / 修改文件 / 配置 / 测试"分组。
每次后续修改都必须追加一节，格式：日期 + 主题 + 逐文件改动 + 动机。

---

## 2026-09-05 — On-policy distillation 三形态（A/B/C）初始实现

### 新增：`tools/distillation/` 离线蒸馏包

| 文件 | 作用 | 关键决定 |
|---|---|---|
| `__init__.py` | 包说明 + 设计契约 | 教师永不进训练进程；CLI 全部带 sidecar 报告 |
| `current_observable.py` | 从 v2 模型文件原样抽出的 `CurrentObservableReportFilterV2`（+`COVAR_V2_VERSION`） | 纯 regex、无 torch；Form A 离线重写与训练期过滤用**同一份**代码，避免双实现漂移 |
| `token_align.py` | 跨词表对齐：教师 token logprob（含 char span）→ 学生 token 分数 | 学生 token 取重叠教师 token 的均值；无重叠时最近邻兜底；返回覆盖率 |
| `rollout_store.py` | 三种 JSONL schema（rollout / scored / preference pair）+ IO + `write_markdown_report` | pair schema 兼容 `tools/preference_rl.load_dpo_pairs_jsonl` |
| `gkd_pool.py` | `GKDPool`（加载/校验/按 round 选最新）、`select`（dense_rl/best_of_k）、`whiten`、`sequence_advantages` | 纯 Python 可单测；`greedy_teacher_scores` 明确用 `teacher_mean_logp`（与逐 token 奖励同尺度），**不是**排序标量 |
| `gkd_torch.py` | `GKDRolloutSubset`（图像 + rollout 张量 + 参考张量 + 教师 token 分数）与 `gkd_collate` | 对齐契约：`label_ids=[t_0..t_{n-1},EOS,PAD..]`，`teacher_token_logps[i]` 监督 `label_ids[i]`；EOS/PAD 位置不入损失 |
| `teacher.py` | `MockTeacher`（确定性冒烟）、`FileTeacher`（外部教师 JSONL 桥接，缺 id 回退 mock 并计数）、`LocalVLMTeacher`（真·多模态教师参考实现）、`build_teacher` 工厂 | mock 的 token 打分按"是否在参考中 + 是否临床词"给分；`vlm:` 惰性导入 transformers（需独立环境） |
| `model_loading.py` | 离线 CLI 的模型加载器 | 按 config `module` 键选择正确子类（修复 `dpo_json.py` 只能加载基线类、会丢 planner 参数的隐患）；`load_student_tokenizer` 从 ckpt_zoo 载入学生 tokenizer |
| `export_rollouts.py` | 导出 greedy + K 条采样 rollout（含学生 logp） | 与部署策略一致地用 v2 `generate`（含时态 bad-words）；教师侧可直接用 `image_path` |
| `teacher_score.py` | `--mode tokens`（B）/ `--mode rank`（C）/ `--mode fill-ref-logps` | rank 输出 chosen/rejected + 奖励 + error_tags；fill-ref-logps 补上 `dpo_json.py add-ref-logps` 的 NotImplemented 缺口 |
| `teacher_rewrite.py` | Form A：教师重写训练报告 → 过滤 → 新 annotation JSON | 只重写 train split；`--max-rewrite` 控预算；空/过短教师输出回退原文并计数 |
| `run_recorder.py` | `RunRecorder`：trial 目录下 `new_lab_run_record.{jsonl,md}` | 白名单只收 `train_/val_/test_` 前缀且排除 time/gpu/mem 等子串 → 记录策略硬编码为"只记算法相关" |

### 修改：`cvt2distilgpt2_mimic_cxr_chen.py`（基线模型）

1. `train_mode` 合法集合 `{ce,dpo,scst}` → `{ce,dpo,scst,gkd}`（gkd 只被
   v2 子类实际使用，但校验发生在基线 `__init__`）。
2. 新参数 `annotation_file="annotation.json"`：`labels_file_path` 改为
   由该参数拼接（Form A 指向重写后的标注，默认行为不变）。
3. 新参数 `run_record=False`：为 True 时构造 `RunRecorder`；新增
   `on_train_epoch_end` 钩子与 `_record_epoch_metrics`，
   `on_validation_epoch_end` / `on_test_epoch_end` 末尾各加一次记录调用。
   全部有 None 保护，不影响旧配置运行。
4. `_training_step_dpo`：当 pair 携带 `reward_chosen/reward_rejected`
   时记录其均值（`train_dpo_reward_chosen/rejected`）——教师奖励随训练
   的演化是 Form C 的核心分析量。

### 修改：`cvt2distilgpt2_mimic_cxr_visual_grounded_v2.py`（COVAR-V2）

1. `CurrentObservableReportFilterV2` 类体（165 行）从本文件移除，改为
   `from tools.distillation.current_observable import ...`；类名在本模块
   继续可用（checkpoint/旧代码不受影响）。
2. `train_mode` 允许 `{ce, gkd, dpo}`（scst 在 COVAR 线上明确不支持）；
   `gkd` 强制要求 `gkd_rollout_path`。
3. 新增 `__init__` 参数：`target_preprocessing`（chen_vocab/clean_only）、
   `gkd_rollout_path / gkd_objective / gkd_weight / gkd_reference_ce_scale /
   gkd_advantage_baseline / gkd_reward_norm / gkd_adv_clip /
   gkd_include_greedy / gkd_best_of_k_min_gap`。
4. `_format_examples_v2`：`target_preprocessing='clean_only'` 时只做
   `clean_report`，不做 Chen 词表截断（教师新词不再被映射成 `<unk>`，
   同时解除 60-token 截断上限；默认值保持旧行为）。
5. `setup()`：`dpo` 模式加载偏好对（此前 v2 的 setup 覆盖把基线的加载
   逻辑丢掉了）；`fit` 且 `gkd` 模式时构造 `GKDRolloutSubset`
   （`_setup_gkd_dataset_v2`），池统计写入 RunRecorder 事件。
6. `train_dataloader` 覆盖：`gkd` 模式返回带 `gkd_collate` 的 DataLoader
   （num_workers=0 时不传 prefetch_factor，顺手规避该已知坑）。
7. `_compute_training_loss_v2` 新增可选 `precomputed_visual_tokens /
   precomputed_plan`：GKD 步内视觉只编码一次，参考分支与 rollout 分支
   共享（默认 None，旧调用不变）。
8. `training_step` 分发 `{ce→原路径, gkd→_training_step_gkd,
   dpo→基线 _training_step_dpo}`（经 v2 forward 覆盖，planner 自动生效）。
9. 新增 `_training_step_gkd`：
   - `best_of_k`：以教师最优 rollout 为目标跑完整 COVAR 损失（稳定对照）。
   - `dense_rl`：rollout 分支逐 token
     `loss = -mean(adv.detach() · logπ_S(y_t))`，advantage 由对齐后的教师
     logprob 经 `sequence_advantages`（序列均值/greedy 基线 + whiten +
     clip）得到；参考分支跑完整 COVAR 损失作锚
     （`total = gkd_weight·gkd + gkd_reference_ce_scale·ref`）。
   - 记录分量：`train_gkd_loss / reward_token_mean / advantage_mean /
     rollout_logp_mean / teacher_score / valid_token_ratio` + COVAR 全部分量。

### 配置（`config/new_lab/`，均基于现有 v2 配置最小差异）

| 文件 | 差异要点 |
|---|---|
| `train_covar_distill_sft.yaml` | `annotation_file: annotation_distilled.json`、`target_preprocessing: clean_only`、`run_record: true`，其余与 v2 训练配置一致 |
| `train_covar_gkd.yaml` | `train_mode: gkd` + gkd_* 参数；`gkd_rollout_path` 指向目录（自动取最新 round）；stage-2 设定 `grounding_warmup_epochs: 0 / ramp 1`、`max_epochs: 4` |
| `train_covar_dpo_distill.yaml` | `train_mode: dpo`、`dpo_reference_free: false`、stage-2 小 LR（1e-6/1e-5）、`max_epochs: 10` |
| `train_baseline_dpo_distill.yaml` | Form C 的纯基线对照（module=chen），monitor 回 `val_chen_cider` |

### 测试与冒烟

- `tests/test_distillation.py`：27 个纯 CPU 测试覆盖过滤器、跨词表对齐、
  池选择/奖励数学（含 stale 对齐防呆）、RunRecorder 白名单、mock/file
  教师、JSONL IO。**不依赖 torch**，任何机器可跑。
- `python -m compileall` 通过（仓库其余文件的既有 SyntaxWarning 未触碰）。

### 未做 / 明确留给后续

- `vlm:` 教师在真实权重上的端到端验证（需要新版 transformers 环境）。
- GKD 的重要性加权（off-policy 修正）：导出时已存
  `student_logp_sum/mean`，留接口未启用。
- `dpo_json.py add-ref-logps` 的 stub 未删除（功能由
  `teacher_score.py --mode fill-ref-logps` 承担），避免动老入口。
- 既有 bug（`train_stage2.yaml` 的 `resumeresume_last` 键、
  `no_vocab.yaml` 命名反转等）不属于本分支范围，未改。
