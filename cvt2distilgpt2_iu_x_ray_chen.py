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
            bert_path='/data/lhy_data/HERGen/google-bert/bert-base-uncased',
            checkpoint_path='/data/lhy_data/MLRG/pretrained-model/chexbert.pth',
            ckpt_dir=self.ckpt_zoo_dir,
            mbatch_size=self.mbatch_size,
            exp_dir=self.exp_dir_trial,
        )
        self.test_chexbert_metrics = CheXbertMetrics(
            bert_path='/data/lhy_data/HERGen/google-bert/bert-base-uncased',
            checkpoint_path='/data/lhy_data/MLRG/pretrained-model/chexbert.pth',
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
        ckpt_name = 'distilgpt2'
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

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """

        with open(self.labels_file_path) as f:
            examples = json.load(f)

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
            i["label"] = i.pop("report")
            i["image_file_path"] = [os.path.join(self.dataset_dir, j) for j in i["image_file_path"]]
            i["label"] = self.chen_tokenizer(i["label"])[:self.chen_max_seq_length]
            i["label"] = self.chen_tokenizer.decode(i["label"][1:])
        return examples

    def encoder_forward(self, images):
        """
        Encoder forward propagation.

        Argument/s:
            images - a mini-batch of images.
            image_batch_ids - batch index for each image.

        Returns:
            encoder_outputs - transformers.modeling_outputs.ModelOutput.
        """
        views = self.multi_input(images)
        image_features = self.encoder(views['images'])['last_hidden_state']
        # # ================= [新增代码开始] =================
        # # 仅在训练时且满足间隔要求时执行，避免拖慢速度
        # save_dir = os.path.join('/data/lhy_data/MIMIC-CXR/invests', "debug_visualizations_IUXRAY")
        # os.makedirs(save_dir, exist_ok=True)
        # global_step = 2
        # save_path = os.path.join(save_dir, f"init_{global_step}.jpg")

        # # --- A. 数据准备 (取 Batch 中的第一个样本) ---
        # # 必须使用 .detach().cpu()
        # img_tensor = images[global_step, 0].detach().cpu()          # shape [3, H, W]
        # feat_tensor = image_features[global_step].detach().cpu()   # shape [Seq_Len, Dim]

        # # --- B. 处理原始图片 (用于显示) ---
        # # 需要反归一化 (假设使用了 ImageNet 的均值和方差)
        # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        # img_show = img_tensor * std + mean
        # # 转换维度为 [H, W, C] 并限制在 [0,1] 范围内供 matplotlib 显示
        # img_show = img_show.permute(1, 2, 0).clamp(0, 1).numpy()

        # # --- C. 处理特征图 (用于制作热力图) ---
        # # CvT 的输出通常是 Sequence 形式，需要reshape回空间尺寸
        # seq_len, dim = feat_tensor.shape
        # if feat_tensor.shape[1] == 576: 
        #     # 情况 A: [Channels, Seq_Len] -> [384, 576]
        #     # 我们需要在 Channel 维度 (dim=0) 上做平均，把 384 个通道压缩成 1 个值
        #     activation = feat_tensor.mean(dim=0) # 结果 shape: [576]
        #     side = int(np.sqrt(576)) # 24
        #     activation_map = activation.view(side, side).numpy() # [24, 24]
                
        # elif feat_tensor.shape[0] == 576:
        #     # 情况 B: [Seq_Len, Channels] -> [576, 384] (防止万一形状反转)
        #     activation = feat_tensor.mean(dim=1) 
        #     side = int(np.sqrt(576))
        #     activation_map = activation.view(side, side).numpy()
            
        # else:
        #     # 情况 C: 可能包含 CLS token (例如 577)
        #     seq_len = max(feat_tensor.shape) # 假设长的那边是序列
        #     if int(np.sqrt(seq_len - 1)) ** 2 == seq_len - 1:
        #         # 有 CLS token，找到它并移除
        #         spatial_dim_idx = 0 if feat_tensor.shape[0] == seq_len else 1
        #         channel_dim_idx = 1 - spatial_dim_idx
                
        #         # 移除 CLS (通常是第0个或最后1个，这里假设第0个)
        #         # 先转成 [Seq, Chan] 方便处理
        #         if spatial_dim_idx == 1:
        #             feat_temp = feat_tensor.permute(1, 0) # 变成 [577, 384]
        #         else:
        #             feat_temp = feat_tensor
                
        #         activation = feat_temp[1:, :].mean(dim=1) # 除去第一个token后求均值
        #         side = int(np.sqrt(seq_len - 1))
        #         activation_map = activation.view(side, side).numpy()
        #     else:
        #         print(f"[Debug Warning] Unknown shape {feat_tensor.shape}, skipping.")
        #         activation_map = None

        # # --- D. 绘图 (只有当 activation_map 成功生成时) ---
        # if activation_map is not None:
        #     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
        #     # 左图：原始图片
        #     axs[0].imshow(img_show)
        #     axs[0].set_title(f"Step {self.global_step}: Input")
        #     axs[0].axis('off')
            
        #     # 右图：特征热力图
        #     # Resize 到原图大小看起来更直观 (可选，这里用插值放大)
        #     import cv2
        #     # 将 24x24 的热力图放大到 384x384 方便叠加对比，或者直接显示 24x24
        #     # 这里直接显示原始分辨率 24x24，能看清具体的激活 Patch
        #     im = axs[1].imshow(activation_map, cmap='jet') 
        #     axs[1].set_title(f"Feature Activation ({activation_map.shape[0]}x{activation_map.shape[1]})")
        #     axs[1].axis('off')
        #     fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

        #     plt.tight_layout()
        #     plt.savefig(save_path, format='jpg', dpi=150, bbox_inches='tight')
        #     plt.close(fig)
        #     print(f"[Debug] Saved visualization to {save_path}")
        image_features = self.encoder_projection(image_features)['projected_encoder_last_hidden_state']
        image_features = self.multi_output(image_features, views['images_per_example'])['last_hidden_state']
        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=image_features)
        return encoder_outputs
 