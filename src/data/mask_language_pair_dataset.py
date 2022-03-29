"""
## mask策略: token span masking

MASS and ProphetNet randomly pick a starting position u in every 64 tokens, and then mask a continuous span from u.
80% of the masked tokens are replaced by [MASK], 10% replaced by random tokens, and 10% unchanged.
The masked length is set to 15% of the total number of tokens.

这里的比例跟bert一致。区别是: 1) continuous span mask, 优势是？方便学习decoder的AR语言模型？  2) 每64token算一个block。

## downstream task

MASS，ProphetNet
"""

import logging
import numpy as np
import torch
import random
import time
import math
from functools import lru_cache

from fairseq import utils
from fairseq.data import data_utils, LanguagePairDataset, LRUCacheDataset
from fairseq.data.language_pair_dataset import collate as _collate
from .noise_util import apply_span_mask

logger = logging.getLogger(__name__)


def collate(
        samples,
        pad_idx,
        eos_idx,
        left_pad_source=True,
        left_pad_target=False,
        input_feeding=True,
):
    """
    相对 fairseq.data.language_pair_dataset.collate 的区别是:
        1. prev_output_tokens的key不再是target，而是 prev_output_tokens（因为自定义了prev_output_tokens，）
        2. 增加了positions（默认position从1开始，mass保留了原句中的位置）
    TODO:
        1. 新key的order问题：
          策略0，全部重写：https://coding.jd.com/alphaw/fairseq_ext/blob/a336c4529822271417fff86a06dcd9f2b0945592/src/data/mask_language_pair_dataset.py
          策略1，继承时也sort一次。前提保证sort结果的不变性。（目前采用该策略，看上去仍然代码冗余）
          策略2：collate增加more_keys参数，或者net_input下的所有都order一遍 （TODO: 先采用策略一）
    """

    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    batch = _collate(samples, pad_idx, eos_idx, left_pad_source=left_pad_source, left_pad_target=left_pad_target,
                     input_feeding=input_feeding)

    # patch
    src_lengths = torch.LongTensor([
        s['source'].ne(pad_idx).long().sum() for s in samples
    ])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    prev_output_positions = merge('prev_output_positions', left_pad=left_pad_target).index_select(0, sort_order)  # 更改
    batch['net_input']['prev_output_positions'] = prev_output_positions  # 更改
    return batch


class MaskedLanguagePairDataset(LanguagePairDataset):
    """ Wrapper for masked language datasets
        (support monolingual and bilingual)

        For monolingual dataset:
        [x1, x2, x3, x4, x5] 
                 ||
                 VV
        src: [x1,  _,  _, x4, x5]
        tgt: [x1, x2] => [x2, x3]   (注意，通常的策略是  [bos, x2] -> [x2, x3])

        default,  _ will be replaced by 8:1:1 (mask, self, rand),
    """

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        return cls(dataset, *args, **kwargs)

    def __init__(
            self, src, src_sizes, src_dict,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True,
            mask_prob=0.15, leave_unmasked_prob=0.1, random_token_prob=0.1,
            mask_whole_words=None,
            block_size=64,
    ):
        super().__init__(src, src_sizes, src_dict,
                         tgt=tgt, tgt_sizes=tgt_sizes, tgt_dict=tgt_dict,
                         left_pad_source=left_pad_source, left_pad_target=left_pad_target,
                         shuffle=shuffle)  # TODO, 是否要加 num_buckets参数
        self.mask_prob = mask_prob
        self.block_size = block_size
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.pred_probs = torch.FloatTensor(
            [1 - leave_unmasked_prob - random_token_prob, leave_unmasked_prob, random_token_prob])

    def __getitem__(self, index):
        """
        TODO: 固定长度，改为dynamic_span_length, dynamic_total_length(AR方式的输入长度也是动态的，遍历1-n)
        decoder能同时看到所有的prev_output_tokens吗？还是AR的方式？
        """
        src_item = self.src[index]
        src_sz = len(src_item)
        # block_dataset, bos may exists inside the sequence, such as 1027,  7570,  7849,  2271, 13091, 102,  7570, ..
        masked_pos = np.array(apply_span_mask(src_sz - 1)) + 1  # start at 1
        target = src_item[masked_pos].clone()
        prev_output_tokens = src_item[masked_pos - 1].clone()
        prev_output_positions = torch.LongTensor(masked_pos)
        src_item[masked_pos] = self.replace(src_item[masked_pos])  # masked_src

        example = {
            'id': index,
            'source': src_item,
            'target': target,
            'prev_output_tokens': prev_output_tokens,  # 注意这里并非简单的shift target
            'prev_output_positions': prev_output_positions,
        }
        return example

    def collater(self, samples):
        return collate(samples, self.src_dict.pad(), self.src_dict.eos())

    def replace(self, x):
        """
        TODO: 这里重写，利用现有的MaskTokensDataset和whole_word_mask等模块
            参考 https://github.com/pytorch/fairseq/blob/master/fairseq/data/mask_tokens_dataset.py#L101
        """
        _x_real = x
        _x_rand = _x_real.clone().random_(self.src_dict.nspecial, len(self.src_dict))
        _x_mask = _x_real.clone().fill_(self.src_dict.mask_index)
        probs = torch.multinomial(self.pred_probs, len(x), replacement=True)
        _x = _x_mask * (probs == 0).long() + \
             _x_real * (probs == 1).long() + \
             _x_rand * (probs == 2).long()
        return _x
