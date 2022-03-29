# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""

TODO:
regression:
  - https://github.com/pytorch/fairseq/blob/1bc83c703ad70d7f62c1e54b197e29b95d07b1f0/fairseq/criterions/sentence_prediction.py#L54


- https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/composite_loss.py
  - 用法：https://github.com/pytorch/fairseq/issues/1175  定义多个模型，多个target

"""

import math
import logging
import numpy as np
import torch
import torch.nn.functional as F

from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion

logger = logging.getLogger(__name__)


@register_criterion('auto_detect')
class AutoCriterion(FairseqCriterion):
    """
    Detect appropriate criterion for your data automatically, it works for both single task training and multi-task joint training.
    It choose criterion automatically for different target,
    e.g. cls_target mlm_target clm_target
    """

    def __init__(self, task, tpu, classification_head_name=None, tagging_head_name=None, apply_label_weight=False):
        super().__init__(task)
        self.tpu = tpu
        self.classification_head_name = classification_head_name
        self.tagging_head_name = tagging_head_name
        self.apply_label_weight = apply_label_weight

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        parser.add_argument('--apply-label-weight',
                            action="store_true",
                            default=False,
                            help='')


    def check_valid(self, masked_tokens):
        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        # https://github.com/pytorch/fairseq/blob/4c55744ec4cb26749cf2cf8dac89942f26ce4bd2/fairseq/criterions/masked_lm.py#L36
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device('cpu'):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )
        return masked_tokens

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = {}    # masked_tokens for different task, there may be more than one task in each batch

        if 'mlm_target' in sample:
            masked_tokens['mlm'] = sample['mlm_target'].ne(self.padding_idx)  # token-level

        if 'cls_target' in sample:
            masked_tokens['cls'] = sample['cls_target'].ge(0)  # sentence-level, negative as invalid, not padding_idx

        if 'multilabel_target' in sample:  # 还需要一个参数表示哪个句子全False
            num_labels = sample['multilabel_target'].size(-1)
            masked_tokens['multilabel'] = torch.sum(sample['multilabel_target'], dim=1).ge(0)  # all inactive as invalid sample, 全0也需要优化。

        if 'clm_target' in sample:  # only part of samples has clm_target, we need decoder_mask to
            decoder_mask = torch.sum(sample['clm_target'].ne(self.padding_idx), dim=1).bool()
            sample['clm_target'] = sample['clm_target'][decoder_mask]
            for k in sample['net_input']:
                if k.startswith('prev_output'):
                    sample['net_input'][k] = sample['net_input'][k][decoder_mask]
            masked_tokens['decoder_mask'] = decoder_mask
            masked_tokens['clm'] = sample['clm_target'].ne(self.padding_idx)

        if 'tag_target' in sample:
            masked_tokens['tag'] = sample['tag_target'].ne(self.padding_idx)
            # masked_tokens['tag'] = sample['tag_target'].ge(0)  # negative as invalid, not padding_idx
            tags = model.get_tag_targets(sample, None) - 1
            kwargs = {
                'tagging_head_name': self.tagging_head_name,
                'tags': tags,
            }
        else:
            kwargs = {}

        # sample_size 不是sentence粒度，而是loss粒度。比如mlm, clm
        sample_sizes = {k: v.int().sum().item() for k, v in masked_tokens.items() if k not in ['decoder_mask']}
        for k, v in sample_sizes.items():
            if v == 0:
                masked_tokens[k] = None
        if 'multilabel' in sample_sizes:   # 多标签分类，每个sample要做num_labels次二分类，
            sample_sizes['multilabel'] *= num_labels

        net_output = model(**sample['net_input'], masked_tokens=masked_tokens,
                           classification_head_name=self.classification_head_name,
                           **kwargs)
        loss = 0
        if sample_sizes.get('mlm', 0) > 0:
            mlm_loss = self._compute_mlm_loss(model, net_output, sample, masked_tokens['mlm'])
            loss += mlm_loss
        if sample_sizes.get('clm', 0) > 0:
            clm_loss = self._compute_clm_loss(model, net_output, sample, masked_tokens['clm'])
            # loss += clm_loss
        if sample_sizes.get('cls', 0) > 0:
            cls_loss, ncorrect_cls = self._compute_cls_loss(model, net_output, sample, masked_tokens['cls'])
            loss += cls_loss
        if sample_sizes.get('multilabel', 0) > 0:
            multilabel_loss, ncorrect_multilabel = self._compute_multilabel_loss(model, net_output, sample, masked_tokens['multilabel'])
            loss += multilabel_loss
        if sample_sizes.get('tag', 0) > 0:
            tag_loss, ncorrect_tag = net_output[1]['tag_out']
            loss += tag_loss
        sample_size = sum(sample_sizes.values())
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,  # avg_loss = loss/sample_size   很关键，负责容易出现loss overflow
        }

        if sample_sizes.get('clm', 0) > 0:
            logging_output['sample_size_clm'] = sample_sizes['clm']
            logging_output['clm_loss']: clm_loss.data if isinstance(clm_loss, torch.Tensor) else clm_loss

        if sample_sizes.get('mlm', 0) > 0:
            logging_output['sample_size_mlm'] = sample_sizes['mlm']
            logging_output['mlm_loss']: mlm_loss.data if isinstance(mlm_loss, torch.Tensor) else mlm_loss

        if sample_sizes.get('multilabel', 0) > 0:
            logging_output['nsentences_multilabel'] = sample_sizes['multilabel'] / num_labels
            logging_output['ncorrect_multilabel'] = ncorrect_multilabel

        if sample_sizes.get('cls', 0) > 0:
            logging_output['sample_size_cls'] = sample_sizes['cls']
            logging_output['ncorrect_cls'] = ncorrect_cls

        if sample_sizes.get('tag', 0) > 0:
            logging_output['sample_size_tag'] = sample_sizes['tag']
            logging_output['ncorrect_tag'] = ncorrect_tag
        return loss, sample_size, logging_output

    def _compute_mlm_loss(self, model, net_output, sample, masked_tokens):
        logits = net_output[1]['mlm_out']
        targets = model.get_mlm_targets(sample, net_output)[masked_tokens]
        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )
        return loss

    def _compute_clm_loss(self, model, net_output, sample, masked_tokens):
        logits = net_output[1]['clm_out']
        targets = model.get_clm_targets(sample, net_output)[masked_tokens]
        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )
        return loss

    def _compute_multilabel_loss(self, model, net_output, sample, masked_tokens):
        """ loss for multilabel classification

        Args:

        - multilabel classification:
          - https://github.com/pytorch/fairseq/issues/2169
          - https://github.com/thunlp/ERNIE/blob/master/code/knowledge_bert/modeling.py#L988  BCEWithLogitsLoss 输入维度是 (*,num_labels)
          - https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/ BCELoss  head仍然是h*num_class维度，采用BCELoss

        注意：
          1. BCEWithLogitsLoss 比 BCELoss 多了个sigmoid
            即：BCEWithLogitsLoss(logits) ==  BCELoss(sigmoid(logits))
          2. pos_weight也很关键。通常多标签的multi-hot，大部分都是0，会造成1的权重下降。
            详见 https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        """
        logits = net_output[1]['multilabel_out']
        targets = model.get_multilabel_targets(sample, net_output)[masked_tokens]
        if not hasattr(self, 'bce_logits_loss'):
            pos_weight = None
            if self.apply_label_weight:
                start_idx = self.task.label_dictionary.nspecial
                num_labels = logits.size(-1)
                label_count = np.array([self.task.label_dictionary.count[i] for i in range(start_idx, start_idx + num_labels)])
                train_size = len(self.task.datasets['train'])
                pos_weight = np.clip((train_size-label_count)/label_count, 0.5, 50)  # 用于处理数据倾斜问题，至多50倍加成
                pos_weight = torch.tensor(pos_weight).to(logits.device)
            self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)
        loss = self.bce_logits_loss(logits, targets.float())
        # ncorrect = torch.sum(logits.gt(0) == targets.gt(0)),  # 跟loss的指标类似，分母是 n_sample * n_labels
        ncorrect = torch.sum(torch.sum(logits.gt(0) != targets.gt(0), dim=1) == 0)  # 用于计算accracy，分母是n_sample
        return loss, ncorrect


    def _compute_cls_loss(self, model, net_output, sample, masked_tokens):
        """

        Args:
            model:
            net_output:  logits
            sample:
            masked_tokens:
        """
        logits = net_output[1]['cls_out']
        targets = model.get_cls_targets(sample, net_output)[masked_tokens]
        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        loss = F.nll_loss(lprobs, targets, reduction='sum')
        preds = logits.argmax(dim=1)
        ncorrect = (preds == targets).sum()
        return loss, ncorrect

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        mlm_loss_sum = sum(log.get('mlm_loss', 0) for log in logging_outputs)
        sample_size_mlm = sum(log.get('sample_size_mlm', 0) for log in logging_outputs)

        clm_loss_sum = sum(log.get('clm_loss', 0) for log in logging_outputs)
        sample_size_clm = sum(log.get('sample_size_clm', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)  # 为什么要 log2？

        if isinstance(mlm_loss_sum, torch.Tensor):
            metrics.log_scalar('mlm_loss', mlm_loss_sum / sample_size_mlm / math.log(2), sample_size_mlm, round=3)
            metrics.log_derived('mlm_ppl', lambda meters: utils.get_perplexity(meters['mlm_loss'].avg))

        if isinstance(clm_loss_sum, torch.Tensor):
            metrics.log_scalar('clm_loss', clm_loss_sum / sample_size_clm / math.log(2), sample_size_clm, round=3)
            metrics.log_derived('clm_ppl', lambda meters: utils.get_perplexity(meters['clm_loss'].avg))

        if len(logging_outputs) > 0 and 'ncorrect_cls' in logging_outputs[0]:
            nsentences = sum(log.get('sample_size_cls', 0) for log in logging_outputs)
            ncorrect_cls = sum(log.get('ncorrect_cls', 0) for log in logging_outputs)
            if nsentences > 0:
                metrics.log_scalar('sent_cls_accuracy', 100.0 * ncorrect_cls / nsentences, nsentences, round=1)

        if len(logging_outputs) > 0 and 'ncorrect_multilabel' in logging_outputs[0]:
            # TODO: 再加个precision recall f1 指标
            # 因为 1. accuracy指标过份严格，跟loss脱节太多  2. accuracy连续性差，跳跃性大（比如模型在收敛，但是accuracy忽高忽低）
            nsentences_multilabel = sum(log.get('nsentences_multilabel', 0) for log in logging_outputs)
            ncorrect_labels = sum(log.get('ncorrect_multilabel', 0) for log in logging_outputs)
            if nsentences_multilabel > 0:
                metrics.log_scalar('sent_multilabel_accuracy', 100.0 * ncorrect_labels / nsentences_multilabel,
                                   nsentences_multilabel, round=1)

        if len(logging_outputs) > 0 and 'ncorrect_tag' in logging_outputs[0]:
            ntags = sum(log.get('sample_size_tag', 0) for log in logging_outputs)
            ncorrect_tag = sum(log.get('ncorrect_tag', 0) for log in logging_outputs)
            if ntags > 0:
                metrics.log_scalar('tag_accuracy', 100.0 * ncorrect_tag / ntags, ntags, round=1)

        # --best-checkpoint-metric accuracy  for model selection, 默认是按照loss

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True



@register_criterion('multitask_lm')
class MultiTaskLMCriterion(AutoCriterion):
    pass
