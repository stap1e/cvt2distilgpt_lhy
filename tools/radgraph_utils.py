import os
import torch
import logging
import torch.nn as nn
import numpy as np

from radgraph.allennlp.commands.predict import _PredictManager
from radgraph.allennlp.common.plugins import import_plugins
from radgraph.allennlp.common.util import import_module_and_submodules
from radgraph.allennlp.predictors.predictor import Predictor
from radgraph.allennlp.models.archival import load_archive
from radgraph.allennlp.common.checks import check_for_gpu

logging.getLogger("radgraph").setLevel(logging.CRITICAL)
logging.getLogger("allennlp").setLevel(logging.CRITICAL)

from radgraph.utils import download_model
from radgraph.utils import (
    preprocess_reports,
    postprocess_reports,
)

from radgraph.rewards import compute_reward
# from appdirs import user_cache_dir
from transformers import AutoTokenizer, AutoModel

# CACHE_DIR = user_cache_dir("radgraph")
CACHE_DIR = r'/mnt/data/liuhongyu/rg/checkpoints'



class RadGraph(nn.Module):
    def __init__(
            self,
            batch_size=1,
            cuda=None,
            model_path=None,
            **kwargs
    ):

        super().__init__()
        if cuda is None:
            cuda = 0 if torch.cuda.is_available() else -1
        self.cuda = cuda
        self.batch_size = batch_size
        self.model_path = model_path

        try:
            if not os.path.exists(self.model_path):
                download_model(
                    repo_id="StanfordAIMI/RRG_scorers",
                    cache_dir=CACHE_DIR,
                    filename="radgraph.tar.gz",
                )
        except Exception as e:
            print("Model download error", e)

        # Model
        import_plugins()
        import_module_and_submodules("radgraph.dygie")

        check_for_gpu(self.cuda)
        archive = load_archive(
            self.model_path,
            weights_file=None,
            cuda_device=self.cuda,
            overrides="",
        )
        self.predictor = Predictor.from_archive(
            archive, predictor_name="dygie", dataset_reader_to_load="validation"
        )

    def forward(self, hyps):

        assert isinstance(hyps, str) or isinstance(hyps, list)
        if isinstance(hyps, str):
            hyps = [hyps]

        hyps = ["None" if not s else s for s in hyps]

        # Preprocessing
        model_input = preprocess_reports(hyps)
        # AllenNLP
        manager = _PredictManager(
            predictor=self.predictor,
            input_file=str(
                model_input
            ),  # trick the manager, make the list as string so it thinks its a filename
            output_file=None,
            batch_size=self.batch_size,
            print_to_console=False,
            has_dataset_reader=True,
        )
        results = manager.run()

        # Postprocessing
        inference_dict = postprocess_reports(results)
        return inference_dict


class F1RadGraph(nn.Module):
    def __init__(
            self,
            reward_level,
            **kwargs
    ):

        super().__init__()
        assert reward_level in ["simple", "partial", "complete", "all"]
        self.reward_level = reward_level
        self.radgraph = RadGraph(**kwargs)

    def forward(self, refs, hyps):
        # Checks
        assert isinstance(hyps, str) or isinstance(hyps, list)
        assert isinstance(refs, str) or isinstance(refs, list)

        if isinstance(hyps, str):
            hyps = [hyps]
        if isinstance(hyps, str):
            refs = [refs]

        assert len(refs) == len(hyps)

        # getting empty report list
        number_of_reports = len(hyps)
        empty_report_index_list = [i for i in range(number_of_reports) if (len(hyps[i]) == 0) or (len(refs[i]) == 0)]
        number_of_non_empty_reports = number_of_reports - len(empty_report_index_list)

        # stacking all reports (hyps and refs)
        report_list = [
                          hypothesis_report
                          for i, hypothesis_report in enumerate(hyps)
                          if i not in empty_report_index_list
                      ] + [
                          reference_report
                          for i, reference_report in enumerate(refs)
                          if i not in empty_report_index_list
                      ]

        assert len(report_list) == 2 * number_of_non_empty_reports

        # getting annotations
        inference_dict = self.radgraph(report_list)

        # Compute reward
        reward_list = []
        hypothesis_annotation_lists = []
        reference_annotation_lists = []
        non_empty_report_index = 0
        for report_index in range(number_of_reports):
            if report_index in empty_report_index_list:
                if self.reward_level == "all":
                    reward_list.append((0., 0., 0.))
                else:
                    reward_list.append(0.)
                continue

            hypothesis_annotation_list = inference_dict[str(non_empty_report_index)]
            reference_annotation_list = inference_dict[
                str(non_empty_report_index + number_of_non_empty_reports)
            ]

            reward_list.append(
                compute_reward(
                    hypothesis_annotation_list,
                    reference_annotation_list,
                    self.reward_level,
                )
            )
            reference_annotation_lists.append(reference_annotation_list)
            hypothesis_annotation_lists.append(hypothesis_annotation_list)
            non_empty_report_index += 1

        assert non_empty_report_index == number_of_non_empty_reports

        if self.reward_level == "all":
            reward_list = ([r[0] for r in reward_list], [r[1] for r in reward_list], [r[2] for r in reward_list])
            mean_reward = (np.mean(reward_list[0]), np.mean(reward_list[1]), np.mean(reward_list[2]))
        else:
            mean_reward = np.mean(reward_list)

        return (
            mean_reward,
            reward_list,
            hypothesis_annotation_lists,
            reference_annotation_lists,
        )

class TextFeatureExtractor(nn.Module):
    """
    专门用于计算 mu_text 的辅助模块
    包含: RadGraph (提取实体) + BERT (提取特征)
    """
    def __init__(self, radgraph_path, bert_path='bert-base-uncased', device='cuda'):
        super().__init__()
        self.device_name = device
        
        # 1. 初始化 RadGraph
        print(f"Loading RadGraph from {radgraph_path}...")
        self.radgraph = RadGraph(model_path=radgraph_path)
        
        # 2. 初始化用于特征提取的 BERT
        print(f"Loading Text Encoder from {bert_path}...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModel.from_pretrained(bert_path)
        
        # 冻结 BERT 参数 (通常计算 mu_text 时不需要更新 BERT，视你的需求而定)
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def forward(self, reports):
        """
        输入: report list (e.g., ["lung is clear", ...])
        输出: mu_text (Batch, Hidden_dim)
        """
        # 使用 RadGraph 获取实体 inference_dict key: '0', '1'...
        inference_dict = self.radgraph(reports)
        # BERT Tokenize padding=True, truncation=True, return_tensors='pt'
        inputs = self.bert_tokenizer(reports, padding=True, truncation=True, 
                                   max_length=512, return_tensors="pt").to(self.bert_model.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask'] # BERT 原生 mask (padding)

        # BERT Embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # [B, L, 768]

        # (Entity Mask)
        batch_size, seq_len = input_ids.shape
        entity_mask = torch.zeros((batch_size, seq_len), device=last_hidden_state.device)

        for idx, report in enumerate(reports):
            # RadGraph 结果
            entities = inference_dict[str(idx)].get('entities', {})
            curr_input_ids = input_ids[idx].tolist()
            
            # 如果没有实体，默认使用 [CLS] token 或者整个句子的 embedding
            if not entities:
                entity_mask[idx] = attention_mask[idx]
                continue

            for _, entity_info in entities.items():
                entity_text = entity_info['tokens'] # string
                entity_token_ids = self.bert_tokenizer.encode(entity_text, add_special_tokens=False) # 将实体文本转为 BERT token ids (不带 special tokens)
                
                # (Sub-sequence Matching)
                len_entity = len(entity_token_ids)
                if len_entity == 0: continue

                # 滑动窗口匹配
                for i in range(len(curr_input_ids) - len_entity + 1):
                    if curr_input_ids[i : i + len_entity] == entity_token_ids:
                        entity_mask[idx, i : i + len_entity] = 1.0
        
        # (Masked Average Pooling)
        mask_expanded = entity_mask.unsqueeze(-1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        
        mu_text = (last_hidden_state * mask_expanded).sum(1) / sum_mask
        
        return mu_text