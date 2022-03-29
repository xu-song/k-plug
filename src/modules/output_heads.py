"""
定义有些模糊，和criterion层有一定的overlap

- SubwordClassificationHead
- MaskedLMHead
- CLSRegressionHead", "CLSClassificationHead

## TODO:
- 借鉴TransformerHeadConfig  https://github.com/NLPatVCU/multitasking_transformers/blob/master/multitasking_transformers/heads/heads.py
- PretrainedConfig https://github.com/huggingface/transformers/blob/master/src/transformers/configuration_utils.py
"""

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    cross_entropy,
)

from .gcn_layer import GraphConvolution

#
# class SentenceClassificationGCNHead(nn.Module):
#     """ https://github.com/tkipf/pygcn/blob/master/pygcn/models.py
#     https://github.com/Megvii-Nanjing/ML-GCN/blob/master/models.py#L75   这里用的glove预训练的label_embedding，
#     """
#
#     def __init__(self, input_dim, inner_dim, num_classes, pooler_activation_fn, pooler_dropout, q_noise=0,
#                  qn_block_size=8, do_spectral_norm=False):
#
#     # def __init__(self, nfeat, nhid, nclass, dropout):
#         super().__init__()
#
#         # self.label_emb =
#
#         label_emb_dim = 512
#
#         self.dense = nn.Linear(input_dim, inner_dim)
#         self.activation_fn = utils.get_activation_fn(pooler_activation_fn)
#         self.dropout = nn.Dropout(p=pooler_dropout)
#
#
#         self.label_emb = nn.Linear(label_emb_dim, num_classes)   # 它起到了 label embedding的作用，Adj矩阵应该作用在这里，
#
#         self.gc1 = GraphConvolution(input_dim, inner_dim)    #   gc2 ( relu( gc1(label_emb ,adj)) , adj)  有维度变换的
#         self.gc2 = GraphConvolution(inner_dim, num_classes)
#
#         self.out2_proj = nn.Linear(inner_dim, num_classes)
#
#
#     def forward(self, features, **kwargs):
#         x = features
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = self.activation_fn(x)
#         x = self.dropout(x)
#
#         self.
#         x = self.out_proj(x)
#         return x

        # x = F.relu(self.gc1(x, adj))  # ML-GCN中采用的 nn.LeakyReLU(0.2)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)





class SentenceClassificationHead(nn.Module):
    """Head for sentence-level classification tasks.
    It also works for MultilabelClassificationHead.
    """

    def __init__(self, input_dim, inner_dim, num_classes, pooler_activation_fn, pooler_dropout, q_noise=0,
                 qn_block_size=8, do_spectral_norm=False):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(pooler_activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

        # https://github.com/pytorch/fairseq/blob/master/fairseq/models/bart/model.py#L302
        if do_spectral_norm:
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        """ take <s> token (equiv. to [CLS])
        Args:
          features: B x C
        Return:
        """
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class SentenceRegressionHead(nn.Module):
    pass


class SentenceRankingHead(nn.Module):
    pass


class MaskedLMHead(nn.Module):
    """Head for Masked Language Modeling. Used for Encoder.
    We apply one more non-linear transformation before the output layer.
    TODO:
        - more general interface, such as VocabPredictionHead.
        - VS CLMHead: there is no dense layer in CLM
    """

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias  # project back to size of vocabulary with bias
        return x




class SequenceTaggingHead(nn.Module):
    """ Head for sequence tagging (token classification) tasks.
    """

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, dropout, use_crf=False):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        self.use_crf = use_crf
        if use_crf:
            from torchcrf import CRF
            self.crf_proj = CRF(num_classes)

    def forward(self, features, masked_tokens=None, tags=None, **kwargs):
        """
        Args:
            features:       (seq_length, batch_size, hidden_dim)
            masked_tokens:  (seq_length, batch_size)
            tags:           (seq_length, batch_size)

        Return:
             nll_loss:
        """
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        ncorrect = None
        if self.use_crf:
            nll_loss = - self.crf_proj(emissions=x, tags=tags, mask=masked_tokens)
        else:
            x = x[masked_tokens]
            tags = tags[masked_tokens]
            nll_loss = cross_entropy(
                x.view(-1, x.size(-1)),
                tags.view(-1),
                reduction='sum',
            )
            preds = x.argmax(dim=1)
            ncorrect = (preds == tags).sum()
        return nll_loss, ncorrect

    def decode(self, features, masked_tokens=None, **kwargs):
        """
        model.decode(emissions)
        Args:
            features:       (seq_length, batch_size, hidden_dim)
            masked_tokens:  (seq_length, batch_size)
        Return:
            pred_tags: [[], [], []]  len(x)=batch_size
        """
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        if self.use_crf:
            pred_tags = self.crf_proj.decode(emissions=x, mask=masked_tokens)
        else:
            pred_tags = []
            x = x.transpose(0, 1)  # bsz,
            masked_tokens = masked_tokens.transpose(0, 1)
            preds = torch.argmax(x[masked_tokens], dim=1)
            preds_iter = iter(list(preds.cpu().numpy()))
            batch_size, seq_length = masked_tokens.shape
            seq_ends = masked_tokens.long().sum(dim=1)
            for idx in range(batch_size):
                pred_tags.append([next(preds_iter) for _ in range(0, seq_ends[idx])])
        return pred_tags


class SequenceTaggingDynamicCRFHead(SequenceTaggingHead):
    """
    Reference:
      https://github.com/pytorch/pytorch/issues/2400
      https://github.com/pytorch/pytorch/issues/11134
    """
    def __init__(self):
        pass
        # from fairseq.modules import DynamicCRF
        # self.crf_layer = DynamicCRF(
        #     num_embedding=len(self.tgt_dict),
        #     low_rank=args.crf_lowrank_approx,  # low_rank
        #     beam_size=args.crf_beam_approx   # important param
        # )
