"""Training/testing stages for COVAR-V2."""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from argparse import Namespace

from dlhpcstarter.trainer import trainer_instance
from dlhpcstarter.utils import (
    get_test_ckpt_path,
    importer,
    load_config_and_update_args,
    resume_from_ckpt_path,
    write_test_ckpt_path,
)
from lightning.pytorch import seed_everything


def stages(args: Namespace):
    args.warm_start_modules = False
    seed_everything(args.trial, workers=True)
    load_config_and_update_args(args)

    TaskModel = importer(
        definition=args.definition,
        module=args.module,
    )
    trainer = trainer_instance(**vars(args))

    if args.train:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        if args.warm_start_ckpt_path:
            model = TaskModel.load_from_checkpoint(
                checkpoint_path=args.warm_start_ckpt_path,
                strict=False,
                **vars(args),
            )
            print(
                "Warm-starting COVAR-V2 non-strictly from: "
                f"{args.warm_start_ckpt_path}"
            )
        elif getattr(args, "warm_start_exp_dir", None):
            warm_dir = os.path.join(
                args.warm_start_exp_dir,
                f"trial_{args.trial}",
            )
            ckpt_path = get_test_ckpt_path(
                warm_dir,
                args.warm_start_monitor,
                args.warm_start_monitor_mode,
                args.test_epoch,
                args.test_ckpt_path,
            )
            model = TaskModel.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                strict=False,
                **vars(args),
            )
            print(
                "Warm-starting COVAR-V2 non-strictly from: "
                f"{ckpt_path}"
            )
        else:
            args.warm_start_modules = True
            model = TaskModel(**vars(args))

        ckpt_path = resume_from_ckpt_path(
            args.exp_dir_trial,
            args.resume_last,
            args.resume_epoch,
            args.resume_ckpt_path,
        )
        trainer.fit(model, ckpt_path=ckpt_path)

    if args.test:
        if args.fast_dev_run:
            if "model" not in locals():
                model = TaskModel(**vars(args))
        else:
            if getattr(args, "other_exp_dir", None):
                other_dir = os.path.join(
                    args.other_exp_dir,
                    f"trial_{args.trial}",
                )
                ckpt_path = get_test_ckpt_path(
                    other_dir,
                    args.other_monitor,
                    args.other_monitor_mode,
                )
            else:
                ckpt_path = get_test_ckpt_path(
                    args.exp_dir_trial,
                    args.monitor,
                    args.monitor_mode,
                    args.test_epoch,
                    args.test_ckpt_path,
                )

            print(f"Testing checkpoint: {ckpt_path}")
            write_test_ckpt_path(
                ckpt_path, args.exp_dir_trial
            )
            model = TaskModel.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                strict=False,
                **vars(args),
            )

        trainer.test(model)
