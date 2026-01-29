import json
import os
from lightning.pytorch import LightningModule

import torch
import transformers
from torchvision import transforms
from transformers.configuration_utils import PretrainedConfig
from cvt2distilgpt2_mimic_cxr_chen import CvT2DistilGPT2MIMICXRChen

from tools.cvt import CvT
from tools.dataset.iu_x_ray_chen import TaskSubset
from tools.dataset.iu_x_ray_chen_tokenizer import TokenizerChen
from tools.encoder_projection import EncoderPermuteProject
from tools.metrics.chexbert import CheXbertMetrics
from tools.metrics.coco import COCOCaptionMetrics
from tools.metrics.report_logger import ReportLogger
from tools.multi_image import MultiImageInput, MultiImageOutput
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from tools.ot_utils import UOTLoss, SelKDLoss
from types import SimpleNamespace
from tools.tta import SKMultiLoss
plt.switch_backend('agg')

class CvT2DistilGPT2IUXRayChen(CvT2DistilGPT2MIMICXRChen):
    def __init__(self,warm_start_modules: bool,exp_dir_trial: str,
            dataset_dir: str,ckpt_zoo_dir: str,mbatch_size: int,
            encoder_lr: float,
            decoder_lr: float,decoder_max_len: int,
            num_test_beams: int,prefetch_factor: int = 5,num_workers: int = 0,
            **kwargs,):
        LightningModule.__init__(self)

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

        # Paths:
        self.labels_file_path = os.path.join(self.dataset_dir, "iu_x-ray_chen", "annotation.json") # 软链接
        self.dataset_dir = os.path.join(self.dataset_dir, "iu_x-ray_chen", "images")
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
            bert_path='/mnt/data/liuhongyu/rg/checkpoints/bert-base-uncased',
            checkpoint_path='/mnt/data/liuhongyu/rg/checkpoints/stanford/chexbert/chexbert.pth',
            ckpt_dir=self.ckpt_zoo_dir,
            mbatch_size=self.mbatch_size,
            exp_dir=self.exp_dir_trial,
        )
        self.test_chexbert_metrics = CheXbertMetrics(
            bert_path='/mnt/data/liuhongyu/rg/checkpoints/bert-base-uncased',
            checkpoint_path='/mnt/data/liuhongyu/rg/checkpoints/stanford/chexbert/chexbert.pth',
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

        # To handle the two views:
        self.multi_input = MultiImageInput()
        self.multi_output = MultiImageOutput()

        # Decoder:
        ckpt_name = 'distilbert/distilgpt2'
        config = transformers.GPT2Config.from_pretrained(
            os.path.join(self.ckpt_zoo_dir, ckpt_name),
            local_files_only=True,
        )
        config.add_cross_attention = True
        config.is_decoder = True

        if self.warm_start_modules:
            decoder = transformers.GPT2LMHeadModel.from_pretrained(            
                os.path.join(self.ckpt_zoo_dir, ckpt_name),
                local_files_only=True,
                config=config,
            )
        else:
            decoder = transformers.GPT2LMHeadModel(config=config)

        # Resize GPT2 embedding to include padding and beginning of sentence token:
        decoder.resize_token_embeddings(config.vocab_size + 2)

        # Decoder tokenizer:
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(
            os.path.join(self.ckpt_zoo_dir, ckpt_name),
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
        class DummyEncoder:
            main_input_name = 'dummy'

            class DummyConfig(PretrainedConfig):
                model_type = 'bert'

            config = DummyConfig()

            def __init__(self, hidden_size):
                self.config.hidden_size = hidden_size
            
            def get_output_embeddings(cls):
                return None
            
            def forward(self):
                return None
            def tie_weights(self):
                '''
                用于兼容 Hugging Face EncoderDecoderModel 的接口检查
                '''
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

        # for otalign
        # self.visual_dummy = nn.Parameter(torch.randn(1, 1, 768))
        # self.uot_loss_fn = UOTLoss(rho=0.1, epsilon=0.1, n_iters=30)
        # self.selkd_loss = SelKDLoss(rho=1.0, epsilon=0.05, n_iters=5)
        self.lambda_uot = 0.
        self.disease_labels_file_path = "/mnt/data/liuhongyu/IUXRay/multi_hot_label.json"
        self.num_diseases = 14
        self.num_classes = 4 # Unmentioned, Positive, Negative, Unclear
        # 也可以是 MLP (Linear -> ReLU -> Linear)
        self.cls_head = nn.Linear(768, self.num_diseases * self.num_classes)

        self.tta_config = SimpleNamespace(
            wsi_sk_weight=0.5,      # Instance Loss Weight (lambda_inst)
            num_heads=5,            # Multi-head numbers
            sk_epsilon=1.0,         # Sinkhorn Entropy regularization (通常设为1或0.1)
            sk_iter_limit=200,       # Sinkhorn 最大迭代次数
            ot_frame='mm',       # OT 计算范围
            sk_type='sppot',     # ["ppot", "sppot", "sppot_stable"]
            rho_base=0.1,           # Curriculum Mass 起始值
            rho_upper=1.0,          # Curriculum Mass 最大值
            gamma_base=1.0,         # KL constraint weight (gamma)
            ema_mm=1.0,
            mm_factor= 0.5,
            mm_iter_limit=100,
        )
        if self.tta_config.wsi_sk_weight > 0:
            self.sk_loss = SKMultiLoss(
                num_heads=self.tta_config.num_heads,
                sk_type=self.tta_config.sk_type,
                ot_frame=self.tta_config.ot_frame,
                sk_iter_limit=self.tta_config.sk_iter_limit,
                epsilon=self.tta_config.sk_epsilon,
            )
            print(f"✅ TTA SKMultiLoss initialized with weight {self.tta_config.wsi_sk_weight}")
        else:
            print(f"❌ TTA SKMultiLoss SKIPPED (Weight={self.tta_config.wsi_sk_weight})")

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """
        with open(self.labels_file_path) as f:
            examples = json.load(f)

        if hasattr(self, "disease_labels_file_path") and self.disease_labels_file_path:
            print(f"Loading disease labels from: {self.disease_labels_file_path}")
            with open(self.disease_labels_file_path, 'r') as f_labels:
                disease_map = json.load(f_labels) 

            def inject_disease_labels(data_list, split_name="unknown"):
                matched_count = 0
                missing_count = 0
                for item in data_list:
                    raw_id = item.get('id')
                    if not raw_id: continue
                    
                    # ID 归一化: 提取纯文件名 (e.g., "1fa79752")
                    clean_id = os.path.splitext(os.path.basename(raw_id))[0]

                    if clean_id in disease_map:
                        item['disease_labels'] = disease_map[clean_id]
                        matched_count += 1
                    else:
                        missing_count += 1
                        # 先填充默认值，具体报错逻辑交给 format_examples 处理
                        item['disease_labels'] = [0] * 14 
                
                print(f"[{split_name}] Disease Labels: {matched_count} matched, {missing_count} missing.")
                if split_name == 'train' and matched_count == 0 and len(data_list) > 0:
                     raise ValueError("Fatal Error: No disease labels matched for TRAIN set! Check ID formats.")

            if "train" in examples: inject_disease_labels(examples["train"], split_name="train")
            if "val" in examples: inject_disease_labels(examples["val"], split_name="val")
            if "test" in examples: inject_disease_labels(examples["test"], split_name="test")

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

        # 3. 分配数据集 (Assign Datasets)
        if stage == "fit" or stage is None:
            self.train_set = TaskSubset(
                examples=self.format_examples(examples["train"], split="train"),
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
                examples=self.format_examples(examples["val"], split="val"),
                tokenizer=self.tokenizer,
                decoder_max_len=self.decoder_max_len,
                colour_space='RGB',
                transforms=self.test_transforms,
                add_bos_eos_manually=True,
            )
            print(f"No. of training & validation examples: {self.train_set.__len__()} & {self.val_set.__len__()}.")

        if stage == "test" or stage is None:
            self.test_set = TaskSubset(
                examples=self.format_examples(examples["test"], split="test"),
                tokenizer=self.tokenizer,
                decoder_max_len=self.decoder_max_len,
                colour_space='RGB',
                transforms=self.test_transforms,
                add_bos_eos_manually=True,
            )
            print(f"No. of test examples: {self.test_set.__len__()}.")

    def format_examples(self, examples, split="train"):
        """
        :param examples: 数据列表
        :param split: 当前处理的数据集名称 ('train', 'val', 'test')
        """
        NUM_LABELS = 14 
        missing_count = 0

        for i in examples:
            if "image_path" in i:
                i["image_file_path"] = i.pop("image_path")
            
            if "disease_labels" in i:
                i["disease_labels"] = torch.tensor(i["disease_labels"], dtype=torch.long)
            else:
                if split == "train":
                    error_id = i.get('id', 'unknown')
                    raise ValueError(f"Fatal Error: Training sample {error_id} is missing 'disease_labels'! Training cannot proceed.")
                else:
                    i["disease_labels"] = torch.zeros(NUM_LABELS, dtype=torch.long)
                    missing_count += 1

            if "report" in i:
                i["label"] = i.pop("report")
            
            if isinstance(i["image_file_path"], list):
                 i["image_file_path"] = [os.path.join(self.dataset_dir, j) for j in i["image_file_path"]]
            else:
                 i["image_file_path"] = os.path.join(self.dataset_dir, i["image_file_path"])

            tokenized = self.chen_tokenizer(i["label"])[:self.chen_max_seq_length]
            i["label"] = self.chen_tokenizer.decode(tokenized[1:]) if len(tokenized) > 1 else ""

        if missing_count > 0:
            print(f"[{split.upper()}] Warning: Filled {missing_count} samples with zero-labels (Normal for inference).")
            
        return examples

    def encoder_forward(self, images, return_feats=False):
        """
        Encoder forward propagation.

        Argument/s:
            images - a mini-batch of images.
            image_batch_ids - batch index for each image.

        Returns:
            encoder_outputs - transformers.modeling_outputs.ModelOutput.
        """
        # img = torch.zeros_like(images)
        # img = torch.ones_like(images)
        views = self.multi_input(images)
        image_features = self.encoder(views['images'])['last_hidden_state']
        image_features = self.encoder_projection(image_features)['projected_encoder_last_hidden_state']
        image_features = self.multi_output(image_features, views['images_per_example'])['last_hidden_state']
        # batch_size = image_features.shape[0]
        # dummy_vector = self.visual_dummy.expand(batch_size, -1, -1)
        # image_features_with_dummy_vector = torch.cat([image_features, dummy_vector], dim=1)
        # c_features = torch.ones_like(image_features) * image_features.min()
        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=image_features)
        if return_feats:
            return encoder_outputs, image_features
        
        return encoder_outputs