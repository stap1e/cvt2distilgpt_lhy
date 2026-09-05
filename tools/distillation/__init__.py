"""On-policy distillation toolkit for the CvT2DistilGPT2/COVAR report generator.

Design contract (new_lab branch):

- All teacher inference happens OFFLINE through standalone CLIs. The training
  process never loads a teacher, so the training GPU is never shared.
- Form A (offline teacher rewrite + SFT): ``teacher_rewrite.py``.
- Form B (on-policy GKD): ``export_rollouts.py`` -> ``teacher_score.py --mode
  tokens`` -> ``train_mode: gkd`` in the COVAR-V2 model.
- Form C (teacher-ranked preference distillation): ``export_rollouts.py`` ->
  ``teacher_score.py --mode rank`` (+ ``--mode fill-ref-logps``) ->
  ``train_mode: dpo``.
- Every CLI writes a sidecar markdown report with algorithm-relevant
  statistics only (no compute/GPU-utilisation logging).

Modules in this package that must stay torch-free at import time so they can
be unit-tested on CPU-only machines: ``current_observable``, ``token_align``,
``rollout_store``, ``gkd_pool``, ``run_recorder``, ``teacher``.
"""
