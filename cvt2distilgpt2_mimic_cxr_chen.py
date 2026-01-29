import json
import os

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
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from tools.ot_utils import UOTLoss, SelKDLoss
from types import SimpleNamespace
from tools.tta import SKMultiLoss
# from tools.mimic_utils import get_disease_embeddings, SemanticOTProjection
plt.switch_backend('agg')

DISEASE_NAMES = [
    "Atelectasis", 
    "Cardiomegaly", 
    "Effusion", 
    "Infiltration", 
    "Mass", 
    "Nodule", 
    "Pneumonia", 
    "Pneumothorax", 
    "Consolidation", 
    "Edema", 
    "Emphysema", 
    "Fibrosis", 
    "Pleural Thickening", 
    "Hernia"
]

class CvT2DistilGPT2MIMICXRChen(LightningModule):
    def __init__(
            self, warm_start_modules: bool, exp_dir_trial: str,
            dataset_dir: str, ckpt_zoo_dir: str, mbatch_size: int,
            encoder_lr: float, decoder_lr: float, decoder_max_len: int,
            num_test_beams: int, prefetch_factor: int = 5, num_workers: int = 0, **kwargs,):
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

        # Paths:
        self.labels_file_path = os.path.join(
            self.dataset_dir,
            "mimic_cxr_chen",
            "annotation.json", #  # 软链接
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
        # self.selkd_loss = SelKDLoss(
        #     rho=1.0, 
        #     epsilon=0.1, 
        #     n_iters=10
        # )
        self.lambda_uot = 0.
        self.disease_labels_file_path = "/mnt/data/liuhongyu/rg/mimic_cxr_chen/mimic_cxr_chexbert_labeled.json"
        self.num_diseases = 14
        self.num_classes = 4 # Unmentioned, Positive, Negative, Unclear
        # 也可以是 MLP (Linear -> ReLU -> Linear)
        self.cls_head = nn.Linear(768, self.num_diseases * self.num_classes)

        # anchors = get_disease_embeddings(
        #     DISEASE_NAMES, 
        #     model_name_or_path=os.path.join(self.ckpt_zoo_dir, ckpt_name), 
        #     device="cpu" 
        # )
        # self.encoder_projection = SemanticOTProjection(
        #     visual_dim=384, 
        #     text_dim=768,   
        #     num_anchors=14,
        #     epsilon=0.05
        # )
        self.tta_config = SimpleNamespace(
            wsi_sk_weight=0.5,      # Instance Loss Weight (lambda_inst)
            num_heads=5,            # Multi-head numbers
            sk_epsilon=1.0,         # Sinkhorn Entropy regularization (通常设为1或0.1)
            sk_iter_limit=50,       # Sinkhorn 最大迭代次数
            ot_frame='batch',       # OT 计算范围
            sk_type='sinkhorn',     # OT 求解器类型
            rho_base=0.1,           # Curriculum Mass 起始值
            rho_upper=1.0,          # Curriculum Mass 最大值
            gamma_base=0.1,         # KL constraint weight (gamma)
        )
        # if self.tta_config.wsi_sk_weight > 0:
        #     self.sk_loss = SKMultiLoss(
        #         num_heads=self.tta_config.num_heads,
        #         sk_type=self.tta_config.sk_type,
        #         ot_frame=self.tta_config.ot_frame,
        #         sk_iter_limit=self.tta_config.sk_iter_limit,
        #         epsilon=self.tta_config.sk_epsilon,
        #         # 如果 SKMultiLoss 内部有 curriculum schedule，可能需要传入 total_iter
        #         # 这里简化处理，假设它能自适应或只需要基础参数
        #     )
        #     print(f"✅ TTA SKMultiLoss initialized with weight {self.tta_config.wsi_sk_weight}")
        # else:
        #     print(f"❌ TTA SKMultiLoss SKIPPED (Weight={self.tta_config.wsi_sk_weight})")

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
                    
                    clean_id = os.path.splitext(os.path.basename(raw_id))[0]

                    if clean_id in disease_map:
                        item['disease_labels'] = disease_map[clean_id]
                        matched_count += 1
                    else:
                        missing_count += 1
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


    def encoder_forward(self, images, return_feats=False):
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
        if return_feats:
            return encoder_outputs, image_features
        return encoder_outputs

    # def forward(self, images, decoder_input_ids, decoder_attention_mask):
    #     """
    #     https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
    #     """
    #     encoder_outputs = self.encoder_forward(images)

    #     # Teacher forcing: labels are given as input
    #     outputs = self.decoder.encoder_decoder(
    #         decoder_input_ids=decoder_input_ids,
    #         decoder_attention_mask=decoder_attention_mask,
    #         encoder_outputs=encoder_outputs,
    #         return_dict=True,
    #     )

    #     return outputs.logits
    
    def forward(self, images, decoder_input_ids, decoder_attention_mask, disease_labels=None, return_feats=False, encoder_outputs=None):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        # 1. 获取 Encoder 输出和视觉特征
        if encoder_outputs is None:
            encoder_outputs, visual_feats = self.encoder_forward(images, return_feats=True)
        else:
            visual_feats = getattr(encoder_outputs, 'last_hidden_state', None)

        # 2. 分类任务 (Classification Head)
        cls_loss = 0.0
        cls_logits = None

        if visual_feats is not None:
            # Global Average Pooling: [Batch, Seq_Len, Dim] -> [Batch, Dim]
            global_visual_feat = visual_feats.mean(dim=1)
            
            # 全连接层: [Batch, 56]
            cls_logits_flat = self.cls_head(global_visual_feat)
            cls_logits = cls_logits_flat.view(-1, self.num_diseases, self.num_classes)

            if disease_labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                # 展平计算: [Batch * 14, 4] vs [Batch * 14]
                cls_loss = loss_fct(
                    cls_logits.view(-1, self.num_classes), 
                    disease_labels.view(-1)
                )
            else:
                print(f"DEBUG: Labels unique values: {torch.unique(disease_labels)}")
                raise ValueError("Multi-hot label activate false!!!!!!")
        else:
            raise ValueError("Multi-hot label activate false!!!!!!")

        # 3. Decoder 前向传播
        req_hidden_states = return_feats or (decoder_input_ids is not None)
        # outputs = self.decoder.encoder_decoder(
        #     decoder_input_ids=decoder_input_ids,
        #     decoder_attention_mask=decoder_attention_mask,
        #     encoder_outputs=encoder_outputs,
        #     return_dict=True,
        #     output_hidden_states=req_hidden_states,
        #     output_attentions=return_feats
        # )
        # text_feats = None
        # if req_hidden_states:
        #     if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        #         text_feats = outputs.hidden_states[-1]
        #     elif hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
        #         text_feats = outputs.decoder_hidden_states[-1]
        #     elif hasattr(outputs, 'last_hidden_state'):
        #         text_feats = outputs.last_hidden_state

        outputs = None
        text_feats = None
        if decoder_input_ids is not None:
            req_hidden_states = True 
            
            outputs = self.decoder.encoder_decoder(
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs, 
                return_dict=True,
                output_hidden_states=req_hidden_states, 
                output_attentions=return_feats
            )
            if req_hidden_states:
                if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
                    text_feats = outputs.decoder_hidden_states[-1]
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    text_feats = outputs.hidden_states[-1]
                else:
                    pass


        ret = {
            'logits': outputs.logits,
            'cls_logits': cls_logits, 
            'cls_loss': cls_loss,     
            'visual_feats': visual_feats,
            'text_feats': text_feats,
        }
        
        return ret
    
    def _two_view_indices(self, sample_tensor, keep_ratio=0.7):
        """
        生成两个视图的随机索引 (View 1 和 View 2)
        sample_tensor: [Batch, N, Dim] 或 [N, Dim]
        """
        if not torch.is_tensor(sample_tensor):
            return None, None
        
        num_patches = sample_tensor.shape[1] 
        
        keep = max(1, int(num_patches * float(keep_ratio)))
        
        # 为整个 Batch 生成相同的掩码模式 (简化计算)，或者为每个样本生成不同掩码
        # 这里为了效率，我们为当前 Batch 生成一组通用的随机排列
        perm = torch.randperm(num_patches, device=sample_tensor.device)
        
        # View 1 的索引
        idx1 = perm[:keep]
        
        # View 2 的索引 (确保有一部分重叠，有一部分不同)
        replace_n = max(1, int(0.2 * keep)) # 20% 的差异
        remain = perm[keep:]
        
        idx2 = idx1.clone()
        if remain.numel() >= replace_n:
            pos = torch.randperm(keep, device=sample_tensor.device)[:replace_n]
            idx2[pos] = remain[:replace_n]
        else:
            idx2 = torch.randperm(num_patches, device=sample_tensor.device)[:keep]
            
        return idx1, idx2
    
    def generate(self, num_beams, images):
        """
        Autoregressively generate a prediction.

        Argument/s:
            num_beams - number of considered beams for the search (one beam is a greedy search).
            images - images for the encoder.

        Returns:
            Indices of the tokens for the predicted sequence.
        """

        encoder_outputs = self.encoder_forward(images)
        # encoder_outputs = torch.ones_like(encoder_outputs)
        # encoder_outputs = torch.zeros_like(encoder_outputs)

        outputs = self.decoder.encoder_decoder.generate(
            # special_token_ids=[self.tokenizer.sep_token_id],
            max_length=self.decoder_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            # mask_token_id=self.tokenizer.pad_token_id,
            num_beams=num_beams,
            return_dict_in_generate=True,
            use_cache=True,
            encoder_outputs=encoder_outputs,
        )

        return outputs['sequences']

    # def training_step(self, batch, batch_idx):
    #     """
    #     https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
    #     """

    #     # Inference:
    #     y_hat = self(
    #         batch['encoder_images'], 
    #         batch['decoder_input_ids'],
    #         batch['decoder_attention_mask'], 
    #     )

    #     # Loss:
    #     loss = F.cross_entropy(
    #         y_hat.permute([0, 2, 1]), batch['label_ids'], ignore_index=self.tokenizer.pad_token_id,
    #     )

    #     # Logging:
    #     self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True, batch_size=y_hat.shape[0])

    #     # Update and log scores for each validation metric:
    #     return loss
    
    def training_step(self, batch, batch_idx):
        images = batch['encoder_images']
        input_ids = batch['decoder_input_ids']
        att_mask = batch['decoder_attention_mask']
        disease_labels = batch.get('disease_labels', None)

        # ---------------- [构造强增强输入 (Strong View)] ----------------
        aug_input_ids = input_ids.clone()
        aug_att_mask = att_mask.clone()

        if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None:
            replace_token_id = self.tokenizer.mask_token_id
        elif hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
            replace_token_id = self.tokenizer.unk_token_id
        else:
            replace_token_id = self.tokenizer.pad_token_id

        aug_input_ids = input_ids.clone()
        mask_prob = 0.1
        rand_matrix = torch.rand(aug_input_ids.shape, device=self.device)
        is_valid_token = (aug_input_ids != self.tokenizer.pad_token_id) & \
                         (aug_input_ids != self.tokenizer.bos_token_id) & \
                         (aug_input_ids != self.tokenizer.eos_token_id)
        mask_indices = (rand_matrix < mask_prob) & is_valid_token
        aug_input_ids[mask_indices] = replace_token_id
        aug_att_mask[mask_indices] = 0
        # rand_mask = torch.bernoulli(torch.full(aug_input_ids.shape, mask_prob, device=self.device)).bool()
        # valid_mask = (aug_input_ids != self.tokenizer.pad_token_id)
        # aug_input_ids[rand_mask & valid_mask] = self.tokenizer.pad_token_id
        # ---------------------------------------------------------------------

        outputs = self.forward(
            images=images, 
            decoder_input_ids=input_ids,
            decoder_attention_mask=att_mask,
            disease_labels=disease_labels
        )
        logits = outputs['logits']
        cls_loss = outputs['cls_loss']
        raw_visual_feats = outputs['visual_feats']

        # 使用 aug_input_ids 进行前向传播
        outputs_aug = self.forward(
            images=images, 
            decoder_input_ids=aug_input_ids, 
            decoder_attention_mask=aug_att_mask, 
            disease_labels=disease_labels
        )
        logits_aug = outputs_aug['logits']
        # ---------------------------------------------------------------------
        ot_loss = 0.0
        if getattr(self, 'sk_loss', None) is not None:
            raw_visual_feats = outputs.get('visual_feats')
            raw_text_feats = outputs.get('text_feats')

            if raw_visual_feats is not None and raw_text_feats is not None:
                # --- Global Representations ---
                feat_visual = raw_visual_feats.mean(dim=1)
                
                # 文本模态: Masked Mean Pooling (处理 Padding)
                if att_mask is not None:
                    mask_expanded = att_mask.unsqueeze(-1).float()
                    sum_embeddings = torch.sum(raw_text_feats * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    feat_text = sum_embeddings / sum_mask
                else:
                    feat_text = raw_text_feats.mean(dim=1)

                # --- Shared Prototype Space ---       
                logits_visual = self.cls_head(feat_visual) # [Batch, Num_Classes]
                logits_text = self.cls_head(feat_text)     # [Batch, Num_Classes]

                # --- SKLoss input ---
                logits_banks = [[logits_visual], [logits_text]]
                feature_bank = [feat_visual, feat_text]
                
                # --- OT Loss ---
                _ot_loss = self.sk_loss(
                    logits_banks, 
                    feature_bank, 
                    similarity_matrix=None, 
                    data_idxs=None
                )
                
                _ot_loss = torch.nan_to_num(_ot_loss, nan=0.0)
                ot_loss = _ot_loss * self.tta_config.wsi_sk_weight
            else:
                raise ValueError("!!!!!! ========== TTA false   1 ========== !!!!!!")
        else:
            raise ValueError("!!!!!! ========== TTA false   2 ========== !!!!!!")
        # =========================================================================


        ce_loss = F.cross_entropy(
            logits.permute(0, 2, 1), 
            batch['label_ids'], 
            ignore_index=self.tokenizer.pad_token_id,
        )

        # ---------------- [Consistency Loss] ----------------
        p_target = F.softmax(logits.detach(), dim=-1) 
        log_p_aug = F.log_softmax(logits_aug, dim=-1)
        consistency_loss = F.kl_div(log_p_aug, p_target, reduction='batchmean')
        # -------------------------------------------------------------------------

        # 4. 总损失融合
        alpha = 0.0  # 分类权重
        # beta = 0.5  # 一致性损失权重
        gamma = 0.0  # OT 损失权重
        
        total_loss = ce_loss + alpha * cls_loss + gamma * ot_loss# + beta * consistency_loss

        # 5. 日志记录
        self.log_dict({
            'train_loss': total_loss,
            'ce_loss': ce_loss,
            'cls_loss': cls_loss,
            # 'cons_loss': consistency_loss, 
            'ot_loss': ot_loss,            
        }, on_step=True, prog_bar=True, on_epoch=True, batch_size=images.shape[0])

        return total_loss

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
