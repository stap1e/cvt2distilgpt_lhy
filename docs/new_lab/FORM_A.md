# Form A：教师重写训练目标 + SFT

## 假设

教师（多模态大模型）看图重写的报告比 Chen 标注质量更高（不截断、
不丢词、语言更规范），且经"当前可观测"过滤后与 COVAR 哲学一致。
仅替换 **train split** 的目标，val/test 参考不动，指标可比。

## 管线

```
annotation.json ──teacher_rewrite.py──> annotation_distilled.json
                     │  (教师重写 → CurrentObservableReportFilterV2)
                     └── annotation_distilled.report.md + .provenance.jsonl
                                            │
                      train_covar_distill_sft.yaml (annotation_file 指向新文件)
```

教师重写后**仍然**经过训练期的时态过滤（幂等，双保险）；
`target_preprocessing: clean_only` 保证教师新词不被映射成 `<unk>`，
并解除 60-token 截断。

## 命令

```bash
# 1) 冒烟（mock=恒等重写，只验证管线与过滤统计）
python -m tools.distillation.teacher_rewrite \
  --annotation-in /data/lhy_data/rg/mimic_cxr_chen/annotation.json \
  --annotation-out /data/lhy_data/rg/mimic_cxr_chen/annotation_distilled_mock.json \
  --config config/new_lab/train_covar_distill_sft.yaml \
  --teacher mock --max-rewrite 200

# 2) 真教师（方式一：外部环境产出 JSONL 后桥接，推荐）
#    外部 JSONL 每行: {"id": <train id>, "rewrite": "<教师重写文本>"}
python -m tools.distillation.teacher_rewrite \
  --annotation-in /data/lhy_data/rg/mimic_cxr_chen/annotation.json \
  --annotation-out /data/lhy_data/rg/mimic_cxr_chen/annotation_distilled.json \
  --config config/new_lab/train_covar_distill_sft.yaml \
  --teacher file:/data/lhy_data/rg/results_new_lab/teacher/rewrites.jsonl

# 2') 真教师（方式二：在带新版 transformers 的独立环境直接跑 vlm 教师）
python -m tools.distillation.teacher_rewrite ... --teacher vlm:google/medgemma-4b-it

# 3) 训练 / 测试
CUDA_VISIBLE_DEVICES=5 dlhpcstarter -t covar_distill_sft \
  -c config/new_lab/train_covar_distill_sft.yaml \
  --stages_module stages_visual_grounded_v2 --train --trial 9223
CUDA_VISIBLE_DEVICES=5 dlhpcstarter -t covar_distill_sft \
  -c config/new_lab/train_covar_distill_sft.yaml \
  --stages_module stages_visual_grounded_v2 --test --trial 9223
```

## 记录清单（每次必查）

| 文件 | 看什么 |
|---|---|
| `annotation_distilled.report.md` | `rewritten_by_teacher` 比例、`temporal_after`（应≈0 但因 fallback 不必严格为 0）、长度分布变化 |
| `.provenance.jsonl` | 抽查 20 条：教师是否引入幻觉、重写是否丢了真阳性发现 |
| `<exp_dir_trial>/new_lab_run_record.md` | train_ce_loss、val_ce_f1_macro / val_chen_cider 逐 epoch |
| 测试输出 | BLEU/CIDEr/ROUGE-L/METEOR + CheXbert F1（macro/micro） |

## 验收与判定

- 主指标：`test CE F1 macro`（配置 monitor 同款）与
  `temporal_claim_rate`（不应劣化）。
- 对照组：同一 seed 的原 v2 训练（相同 epoch 预算下取各自最优 ckpt）。
- 若 CIDEr 上升但 CE F1 下降 → 教师重写偏向"流畅但事实松"，优先检查
  provenance 里阴性发现是否被改写丢失，再考虑重写 prompt 加约束。

## 风险

- 教师幻觉直接进训练目标（A 形态无学生在环纠偏）——先跑小样本
  人工抽查再全量。
- `clean_only` 使目标变长，`mean_generated_tokens` 与 EOS 校准行为都会
  变化；run record 里对照基线的这两个量。
