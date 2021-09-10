# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch
import torch.nn.functional as F

from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('sequence_tagging')
class SequenceTaggingLoss(FairseqCriterion):

    def __init__(self, task, tpu, tagging_head_name=None):
        super().__init__(task)
        self.tpu = tpu
        self.tagging_head_name = tagging_head_name
        self.tag_offset = 1  # offset for [PAD]   TODO: use masked_tokens instead

    def forward(self, model, sample, reduce=True):
        masked_tokens = sample['target'].ne(self.padding_idx).transpose(0, 1)
        tags = (sample['target'] - self.tag_offset).transpose(0, 1)
        nll_loss, ncorrect = model(**sample['net_input'], masked_tokens=masked_tokens,
                                   tagging_head_name=self.tagging_head_name, tags=tags)
        loss = nll_loss
        sample_size = sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'ncorrect': ncorrect,
            'sample_size': sample_size
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

        if len(logging_outputs) > 0 and logging_outputs[0].get('ncorrect', None) is not None:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            if sample_size > 0:
                metrics.log_scalar('accuracy', 100.0 * ncorrect / sample_size, sample_size, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
