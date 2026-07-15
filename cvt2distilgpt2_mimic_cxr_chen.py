import json
import os
from typing import Optional

import torch
import torch.nn.functional as F
import transformers
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers.configuration_utils import PretrainedConfig

from tools.cvt import CvT
from tools.dataset.mimc_cxr_chen import TaskSubset
from tools.dataset.mimic_cxr_chen_tokenizer import TokenizerChen
from tools.encoder_projection import EncoderPermuteProject
from tools.metrics.chexbert import CheXbertMetrics
from tools.metrics.coco import COCOCaptionMetrics
from tools.metrics.report_logger import ReportLogger
from tools.preference_rl import (
    build_decoder_lm_batch,
    compute_dpo_loss,
    load_dpo_pairs_jsonl,
    sequence_logprobs_from_logits,
)
from tools.reward_evaluator import LiteClinicalRewardEvaluator


class CvT2DistilGPT2MIMICXRChen(LightningModule):
    def __init__(
            self,
            warm_start_modules: bool,
            exp_dir_trial: str,
            dataset_dir: str,
            ckpt_zoo_dir: str,
            mbatch_size: int,
            encoder_lr: float,
            decoder_lr: float,
            decoder_max_len: int,
            num_test_beams: int,
            prefetch_factor: int = 5,
            num_workers: int = 0,
            use_chen_vocab_preprocessing: bool = False,
            train_mode: str = "ce",
            dpo_pair_path: Optional[str] = None,
            dpo_beta: float = 0.1,
            dpo_ce_weight: float = 0.1,
            dpo_reference_free: bool = False,
            dpo_missing_pair_policy: str = "ce",
            rl_weight: float = 0.02,
            rl_ce_weight: float = 0.1,
            rl_sample_temperature: float = 0.8,
            rl_top_p: float = 0.9,
            rl_num_samples: int = 1,
            reward_min_len: int = 5,
            reward_max_len: int = 120,
            reward_repeat_ngram: int = 3,
            reward_short_penalty: float = 0.15,
            reward_repeat_penalty: float = 0.2,
            reward_normal_template_penalty: float = 0.1,
            **kwargs,
    ):
        super().__init__()

        self.warm_start_modules = warm_start_modules
        self.exp_dir_trial = exp_dir_trial
        self.dataset_dir = dataset_dir
        self.ckpt_zoo_dir = ckpt_zoo_dir
        self.mbatch_size = mbatch_size
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.decoder_max_len = decoder_max_len
        self.num_test_beams = num_test_beams
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.use_chen_vocab_preprocessing = bool(use_chen_vocab_preprocessing)
        self.train_mode = train_mode
        self.dpo_pair_path = dpo_pair_path
        self.dpo_beta = dpo_beta
        self.dpo_ce_weight = dpo_ce_weight
        self.dpo_reference_free = dpo_reference_free
        self.dpo_missing_pair_policy = dpo_missing_pair_policy
        self.rl_weight = rl_weight
        self.rl_ce_weight = rl_ce_weight
        self.rl_sample_temperature = rl_sample_temperature
        self.rl_top_p = rl_top_p
        self.rl_num_samples = rl_num_samples
        self.reward_min_len = reward_min_len
        self.reward_max_len = reward_max_len
        self.reward_repeat_ngram = reward_repeat_ngram
        self.reward_short_penalty = reward_short_penalty
        self.reward_repeat_penalty = reward_repeat_penalty
        self.reward_normal_template_penalty = reward_normal_template_penalty
        self.dpo_pairs = {}
        self._dpo_reference_free_warned = False

        if self.train_mode not in {"ce", "dpo", "scst"}:
            raise ValueError(f'Unsupported train_mode: {self.train_mode}. Expected "ce", "dpo", or "scst".')
        if self.dpo_missing_pair_policy not in {"ce", "skip"}:
            raise ValueError(
                f'Unsupported dpo_missing_pair_policy: {self.dpo_missing_pair_policy}. '
                f'Expected "ce" or "skip".'
            )
        self.reward_evaluator = LiteClinicalRewardEvaluator(
            min_len=self.reward_min_len,
            max_len=self.reward_max_len,
            repeat_ngram=self.reward_repeat_ngram,
            short_penalty=self.reward_short_penalty,
            repeat_penalty=self.reward_repeat_penalty,
            normal_template_penalty=self.reward_normal_template_penalty,
        )

        # Paths:
        self.labels_file_path = os.path.join(
            self.dataset_dir,
            "mimic_cxr_chen",
            "annotation.json",
        )
        self.dataset_dir = os.path.join(
            self.dataset_dir,
            "mimic_cxr_chen",
            "mimic_cxr_jpg",
        )
        self.chen_tokenizer = TokenizerChen(
            ann_path=self.labels_file_path,
            threshold=3,
        )
        self.chen_max_seq_length = 60

        """
        Evaluation metrics
        
        These need to be defined correctly in order for them to be placed on the correct device:
        https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#torchmetrics-in-pytorch-lightning
        """      
        self.val_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge"])
        self.test_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "meteor", "rouge"])

        # CheXbert classification metrics:
        self.val_chexbert_metrics = CheXbertMetrics(
            bert_path='bert-base-uncased',
            checkpoint_path='stanford/chexbert/chexbert.pth',
            ckpt_dir=self.ckpt_zoo_dir,
            mbatch_size=self.mbatch_size,
            exp_dir=self.exp_dir_trial,
        )
        self.test_chexbert_metrics = CheXbertMetrics(
            bert_path='bert-base-uncased',
            checkpoint_path='stanford/chexbert/chexbert.pth',
            ckpt_dir=self.ckpt_zoo_dir,
            mbatch_size=self.mbatch_size,
            exp_dir=self.exp_dir_trial,
        )

        # Report logging:
        self.val_report_logger = ReportLogger(exp_dir=self.exp_dir_trial, split='val_reports')
        self.test_report_logger = ReportLogger(exp_dir=self.exp_dir_trial, split='test_reports')

        # Encoder:
        self.encoder = CvT(
            warm_start=self.warm_start_modules,
            model_config='cvt-21-384x384',
            ckpt_name='CvT-21-384x384-IN-22k',
            ckpt_dir=self.ckpt_zoo_dir,
            is_encoder=True,
        )
        self.encoder_projection = EncoderPermuteProject(
            permute_encoder_last_hidden_state=[0, 2, 1],
            encoder_last_hidden_state_size=384,
            decoder_hidden_state_size=768,
        )

        # Decoder:
        ckpt_name = 'distilbert/distilgpt2'
        decoder_ckpt_path = os.path.join(self.ckpt_zoo_dir, ckpt_name)
        if not os.path.isdir(decoder_ckpt_path):
            raise FileNotFoundError(
                f'Local decoder checkpoint directory not found: {decoder_ckpt_path}. '
                f'Please download distilgpt2 into this directory before running.'
            )

        config = transformers.GPT2Config.from_pretrained(
            decoder_ckpt_path,
            local_files_only=True,
        )
        config.add_cross_attention = True
        config.is_decoder = True

        if self.warm_start_modules:
            decoder = transformers.GPT2LMHeadModel.from_pretrained(
                decoder_ckpt_path,
                local_files_only=True,
                config=config,
            )
        else:
            decoder = transformers.GPT2LMHeadModel(config=config)

        # Resize GPT2 embedding to include padding and beginning of sentence token:
        decoder.resize_token_embeddings(config.vocab_size + 2)

        # Decoder tokenizer:
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(
            decoder_ckpt_path,
            local_files_only=True,
        )
        self.tokenizer.add_special_tokens({"bos_token": "[BOS]", 'pad_token': '[PAD]'})

        # Print the special tokens:
        print('Description, Special token, Index')
        for k, v in self.tokenizer.special_tokens_map.items():
            if k != 'additional_special_tokens':
                print(f'{k}, {v}, {getattr(self.tokenizer, k + "_id")}')
            else:
                for i, j in zip(self.tokenizer.additional_special_tokens, self.tokenizer.additional_special_tokens_ids):
                    print(f'additional_special_token, {i}, {j}')

        # We don't actually want to use the encoder of the EncoderDecoderModel, create a dummy encoder:
        class DummyEncoder(torch.nn.Module):
            main_input_name = 'dummy'

            class DummyConfig(PretrainedConfig):
                model_type = 'bert'

            def __init__(self, hidden_size):
                super().__init__()
                self.config = self.DummyConfig()
                self.config.hidden_size = hidden_size

            def get_output_embeddings(self):
                return None

            def forward(self):
                return None

            def tie_weights(self):
                pass

            def _init_weights(self, module):
                pass

        # Use Hugging Face Transformers EncoderDecoderModel to generate conditionally:
        dummy_encoder = DummyEncoder(hidden_size=decoder.config.hidden_size)

        # To be compatible with previous the framework (and hence, the available checkpoint):
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_decoder = transformers.EncoderDecoderModel(encoder=dummy_encoder, decoder=decoder)
        self.decoder = Decoder()

        # Image transformations:
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(size=384 + 64),
                transforms.RandomCrop(
                    size=[384, 384],
                    pad_if_needed=True,
                ),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize(size=384 + 64),
                transforms.CenterCrop(size=[384, 384]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """

        with open(self.labels_file_path) as f:
            examples = json.load(f)

        if self.train_mode == "dpo":
            if self.dpo_pair_path is None:
                raise ValueError(
                    "train_mode='dpo' requires dpo_pair_path. "
                    "Provide a DPO JSONL file with id/chosen/rejected pairs, or use train_mode='ce'."
                )
            self.dpo_pairs = load_dpo_pairs_jsonl(self.dpo_pair_path)

        # Dataset statistics:
        images = set()
        for i in examples["train"]:
            images.update(i["image_path"])
        print(
            "Training set #images: {}, #studies: {}".format(
                len(images), len(examples["train"])
            )
        )

        images = set()
        for i in examples["val"]:
            images.update(i["image_path"])
        print(
            "Validation set #images: {}, #studies: {}".format(
                len(images), len(examples["val"])
            )
        )

        images = set()
        for i in examples["test"]:
            images.update(i["image_path"])
        print(
            "Test set #images: {}, #studies: {}".format(
                len(images), len(examples["test"])
            )
        )

        # Assign train & validation sets:
        if stage == "fit" or stage is None:
            self.train_set = TaskSubset(
                examples=self.format_examples(examples["train"]),
                tokenizer=self.tokenizer,
                decoder_max_len=self.decoder_max_len,
                colour_space='RGB',
                transforms=self.train_transforms,
                self_critical=False,
                train=True,
                add_bos_eos_manually=True,
                num_samples=None,
            )

            self.val_set = TaskSubset(
                examples=self.format_examples(examples["val"]),
                tokenizer=self.tokenizer,
                decoder_max_len=self.decoder_max_len,
                colour_space='RGB',
                transforms=self.test_transforms,
                add_bos_eos_manually=True,
            )
            print(
                "No. of training & validation examples: {} & {}.".format(
                    self.train_set.__len__(), self.val_set.__len__()
                )
            )

        # Assign test set:
        if stage == "test" or stage is None:
            self.test_set = TaskSubset(
                examples=self.format_examples(examples["test"]),
                tokenizer=self.tokenizer,
                decoder_max_len=self.decoder_max_len,
                colour_space='RGB',
                transforms=self.test_transforms,
                add_bos_eos_manually=True,
            )
            print(
                "No. of test examples: {}.".format(
                    self.test_set.__len__()
                )
            )

    def format_examples(self, examples):
        for i in examples:
            i["image_file_path"] = i.pop("image_path")
            report = i.pop("report")
            i["image_file_path"] = [os.path.join(self.dataset_dir, j) for j in i["image_file_path"]]
            if self.use_chen_vocab_preprocessing:
                token_ids = self.chen_tokenizer(report)[:self.chen_max_seq_length]
                report = self.chen_tokenizer.decode(token_ids[1:])
            else:
                report = self.chen_tokenizer.clean_report(report)
            i["label"] = report
        return examples

    def train_dataloader(self, shuffle=True):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        return DataLoader(
            self.train_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#val-dataloader
        """
        return DataLoader(
            self.val_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-dataloader
        """
        return DataLoader(
            self.test_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
        )
    
    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        grouped_parameters = [
            {"params": self.encoder.parameters(), 'lr': self.encoder_lr},
            {"params": self.encoder_projection.parameters(), 'lr': self.decoder_lr},
            {"params": self.decoder.parameters(), 'lr': self.decoder_lr},
        ]

        optimiser = {'optimizer': torch.optim.AdamW(grouped_parameters, lr=self.decoder_lr)}
        return optimiser


    def encoder_forward(self, images):
        """
        Encoder forward propagation.

        Argument/s:
            images - a mini-batch of images.
            image_batch_ids - batch index for each image.

        Returns:
            encoder_outputs - transformers.modeling_outputs.ModelOutput.
        """
        image_features = self.encoder(images)['last_hidden_state']
        image_features = self.encoder_projection(image_features)['projected_encoder_last_hidden_state']
        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=image_features)
        return encoder_outputs

    def forward(self, images, decoder_input_ids, decoder_attention_mask):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        encoder_outputs = self.encoder_forward(images)

        # Teacher forcing: labels are given as input
        outputs = self.decoder.encoder_decoder(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )

        return outputs.logits

    def generate(
            self,
            num_beams,
            images,
            do_sample: bool = False,
            top_p: Optional[float] = None,
            temperature: Optional[float] = None,
            num_return_sequences: int = 1,
    ):
        """
        Autoregressively generate a prediction.

        Argument/s:
            num_beams - number of considered beams for the search (one beam is a greedy search).
            images - images for the encoder.

        Returns:
            Indices of the tokens for the predicted sequence.
        """

        encoder_outputs = self.encoder_forward(images)

        generate_kwargs = {
            'max_length': self.decoder_max_len,
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'num_beams': num_beams,
            'do_sample': do_sample,
            'num_return_sequences': num_return_sequences,
            'return_dict_in_generate': True,
            'use_cache': True,
            'encoder_outputs': encoder_outputs,
        }
        if top_p is not None:
            generate_kwargs['top_p'] = top_p
        if temperature is not None:
            generate_kwargs['temperature'] = temperature

        outputs = self.decoder.encoder_decoder.generate(**generate_kwargs)

        return outputs['sequences']

    def _compute_ce_loss(self, batch):
        # Inference:
        y_hat = self(
            batch['encoder_images'],
            batch['decoder_input_ids'],
            batch['decoder_attention_mask'],
        )

        # Loss:
        loss = F.cross_entropy(
            y_hat.permute([0, 2, 1]), batch['label_ids'], ignore_index=self.tokenizer.pad_token_id,
        )
        return loss, y_hat

    def _training_step_ce(self, batch, batch_idx):
        ce_loss, y_hat = self._compute_ce_loss(batch)
        self.log_dict(
            {'train_loss': ce_loss, 'train_ce_loss': ce_loss},
            on_step=True,
            on_epoch=True,
            batch_size=y_hat.shape[0],
        )
        return ce_loss

    def _dpo_reference_logps(self, pairs, device):
        ref_chosen_logps = []
        ref_rejected_logps = []
        for pair in pairs:
            has_ref_logps = 'ref_logp_chosen' in pair and 'ref_logp_rejected' in pair
            if has_ref_logps:
                ref_chosen_logps.append(float(pair['ref_logp_chosen']))
                ref_rejected_logps.append(float(pair['ref_logp_rejected']))
            elif self.dpo_reference_free:
                if not self._dpo_reference_free_warned:
                    print(
                        'Using reference-free DPO because ref_logp_chosen/ref_logp_rejected '
                        'are missing. This is intended for smoke tests, not formal experiments.'
                    )
                    self._dpo_reference_free_warned = True
                ref_chosen_logps.append(0.0)
                ref_rejected_logps.append(0.0)
            else:
                raise RuntimeError(
                    'DPO pair is missing ref_logp_chosen/ref_logp_rejected. '
                    'Precompute reference log-probs in dpo_pair_path or set dpo_reference_free=True.'
                )

        return (
            torch.tensor(ref_chosen_logps, dtype=torch.float, device=device),
            torch.tensor(ref_rejected_logps, dtype=torch.float, device=device),
        )

    def _training_step_dpo(self, batch, batch_idx):
        ce_loss, ce_logits = self._compute_ce_loss(batch)
        device = batch['encoder_images'].device

        chosen_texts = []
        rejected_texts = []
        pair_rows = []
        image_indices = []

        for i, example_id in enumerate(batch['id']):
            pair = self.dpo_pairs.get(str(example_id))
            if pair is None:
                continue
            if 'rejected' not in pair:
                raise ValueError(f'DPO pair for id={example_id} is missing rejected text.')

            chosen_texts.append(pair.get('chosen') or batch['labels'][i])
            rejected_texts.append(pair['rejected'])
            pair_rows.append(pair)
            image_indices.append(i)

        if not pair_rows:
            self.log_dict(
                {
                    'train_loss': ce_loss,
                    'train_ce_loss': ce_loss,
                    'train_total_loss': ce_loss,
                    'dpo_pair_coverage': torch.tensor(0.0, device=device),
                },
                on_step=True,
                on_epoch=True,
                batch_size=ce_logits.shape[0],
            )
            return ce_loss

        image_indices = torch.tensor(image_indices, dtype=torch.long, device=device)
        dpo_images = batch['encoder_images'].index_select(0, image_indices)

        chosen_batch = build_decoder_lm_batch(self.tokenizer, chosen_texts, self.decoder_max_len, device)
        rejected_batch = build_decoder_lm_batch(self.tokenizer, rejected_texts, self.decoder_max_len, device)

        chosen_logits = self(
            dpo_images,
            chosen_batch['decoder_input_ids'],
            chosen_batch['decoder_attention_mask'],
        )
        rejected_logits = self(
            dpo_images,
            rejected_batch['decoder_input_ids'],
            rejected_batch['decoder_attention_mask'],
        )

        policy_chosen_logps = sequence_logprobs_from_logits(
            chosen_logits,
            chosen_batch['label_ids'],
            self.tokenizer.pad_token_id,
        )
        policy_rejected_logps = sequence_logprobs_from_logits(
            rejected_logits,
            rejected_batch['label_ids'],
            self.tokenizer.pad_token_id,
        )
        ref_chosen_logps, ref_rejected_logps = self._dpo_reference_logps(pair_rows, device)

        dpo_loss, dpo_metrics = compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            self.dpo_beta,
        )
        total_loss = dpo_loss + self.dpo_ce_weight * ce_loss
        coverage = len(pair_rows) / max(1, len(batch['id']))

        log_values = {
            'train_loss': total_loss,
            'train_total_loss': total_loss,
            'train_dpo_loss': dpo_loss,
            'train_ce_loss': ce_loss,
            'dpo_pair_coverage': torch.tensor(coverage, device=device),
        }
        log_values.update(dpo_metrics)
        self.log_dict(log_values, on_step=True, on_epoch=True, batch_size=ce_logits.shape[0])
        return total_loss

    def _training_step_scst(self, batch, batch_idx):
        if self.rl_num_samples != 1:
            raise NotImplementedError('SCST currently supports rl_num_samples=1.')

        ce_loss, ce_logits = self._compute_ce_loss(batch)
        device = batch['encoder_images'].device

        with torch.no_grad():
            greedy_ids = self.generate(1, batch['encoder_images'])
            sampled_ids = self.generate(
                1,
                batch['encoder_images'],
                do_sample=True,
                top_p=self.rl_top_p,
                temperature=self.rl_sample_temperature,
                num_return_sequences=self.rl_num_samples,
            )

        greedy_reports = self.tokenizer.batch_decode(greedy_ids, skip_special_tokens=True)
        sampled_reports = self.tokenizer.batch_decode(sampled_ids, skip_special_tokens=True)

        sample_scores = self.reward_evaluator.score_batch(sampled_reports, references=batch['labels'])
        greedy_scores = self.reward_evaluator.score_batch(greedy_reports, references=batch['labels'])
        reward_sample = torch.tensor([i['reward'] for i in sample_scores], dtype=torch.float, device=device)
        reward_greedy = torch.tensor([i['reward'] for i in greedy_scores], dtype=torch.float, device=device)
        advantage = reward_sample - reward_greedy

        sampled_batch = build_decoder_lm_batch(self.tokenizer, sampled_reports, self.decoder_max_len, device)
        sampled_logits = self(
            batch['encoder_images'],
            sampled_batch['decoder_input_ids'],
            sampled_batch['decoder_attention_mask'],
        )
        sampled_logps = sequence_logprobs_from_logits(
            sampled_logits,
            sampled_batch['label_ids'],
            self.tokenizer.pad_token_id,
        )

        scst_loss = -(advantage.detach() * sampled_logps).mean()
        total_loss = self.rl_weight * scst_loss + self.rl_ce_weight * ce_loss
        self.log_dict(
            {
                'train_loss': total_loss,
                'train_total_loss': total_loss,
                'train_scst_loss': scst_loss,
                'train_reward_sample': reward_sample.mean(),
                'train_reward_greedy': reward_greedy.mean(),
                'train_advantage': advantage.mean(),
                'train_ce_loss': ce_loss,
            },
            on_step=True,
            on_epoch=True,
            batch_size=ce_logits.shape[0],
        )
        return total_loss

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """
        if self.train_mode == "ce":
            return self._training_step_ce(batch, batch_idx)
        elif self.train_mode == "dpo":
            return self._training_step_dpo(batch, batch_idx)
        elif self.train_mode == "scst":
            return self._training_step_scst(batch, batch_idx)
        else:
            raise ValueError(f'Unsupported train_mode: {self.train_mode}.')

    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """

        # Greedy search:
        output_ids = self.generate(1, batch['encoder_images'])

        # Findings and impression sections:
        generated = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # Log reports:
        self.val_report_logger.update(generated, dicom_ids=batch['id'])

        # Evaluate:
        self.val_chexbert_metrics.update(generated, batch['labels'], ids=batch['id'])
        self.val_coco_metrics.update(generated, [[i] for i in batch['labels']], ids=batch['id'])

    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-validation-epoch-end
        """
        # Save reports:
        self.val_report_logger.compute(self.current_epoch)
        self.val_report_logger.reset()

        scores = {}

        output = self.val_chexbert_metrics.compute()
        scores.update(output)
        self.val_chexbert_metrics.reset()

        output = self.val_coco_metrics.compute()
        scores.update(output)
        self.val_coco_metrics.reset()

        self.log_dict({f'val_{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """

        # Beam search:
        output_ids = self.generate(self.num_test_beams, batch['encoder_images'])

        # Generated report:
        generated = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # Log reports:
        self.test_report_logger.update(generated, dicom_ids=batch['id'])

        # Evaluate:
        self.test_chexbert_metrics.update(generated, batch['labels'], ids=batch['id'])
        self.test_coco_metrics.update(generated, [[i] for i in batch['labels']], ids=batch['id'])

    def on_test_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-test-epoch-end
        """

        # Save reports:
        self.test_report_logger.compute(self.current_epoch)
        self.test_report_logger.reset()

        scores = {}

        output = self.test_chexbert_metrics.compute()
        scores.update(output)
        self.test_chexbert_metrics.reset()

        output = self.test_coco_metrics.compute()
        scores.update(output)
        self.test_coco_metrics.reset()

        self.log_dict({f'test_{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)
