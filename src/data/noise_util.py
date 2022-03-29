# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
- 中文混淆集(易错、同音、同形)。用于MLM，目标函数采用 1正多负的pairwise ranking策略。
- 插入
- 交换


## Reference

- https://github.com/pytorch/fairseq/blob/master/fairseq/data/noising.py   semisupervised_translation
- https://github.com/pytorch/fairseq/blob/master/fairseq/data/mask_tokens_dataset.py  for BERT Roberta vq-wav2vec
- https://github.com/pytorch/fairseq/blob/master/fairseq/data/denoising_dataset.py

"""


import random
import math
import numpy as np

def apply_pos_noise():
    pass


def apply_mask_dropout(encoder_padding_mask):
    """ A B C D E --> B D E
    drop encoder_out randomly
    refactor: pooler? dropout? random_mask? corrupt? noise?
    Reference：
      - https://github.com/pytorch/fairseq/blob/master/tests/test_inference_dropout.py
    """
    pooled_mask = None
    return pooled_mask


def apply_teacher_forcing_dropout():
    """ mask
    randomly zeroes some of the elements of the input tensor with probability
    """
    pass


def apply_entity_mask_for_mlm(src_len, src_entity, mask_prob=0.1, ignore_index=set(), max_len=512):
    """
    apply entity mask for Masked Language Model

    Args:
        src_entity: entity positions of source tokens
          [[ent1_start, ent1_end], [ent2_start, ent2_end], ...]

    """
    if len(src_entity) == 0:
        return []

    # 1. excludes
    valid_idx = []
    for idx in range(len(src_entity)):
        ent_id = set(range(src_entity[idx, 0], src_entity[idx, 1]))
        if ent_id & ignore_index:
            continue
        if src_entity[idx, 1] > max_len - 2:
            continue
        valid_idx.append(idx)
    valid_entity = src_entity[valid_idx, :]
    valid_cnt = len(valid_entity)
    if valid_cnt == 0:
        return []

    # 2. apply entity mask
    entity_size = valid_entity[:, 1] - valid_entity[:, 0]
    mask_token_cnt = (src_len - len(ignore_index)) * mask_prob
    mask_entity_cnt = math.ceil(mask_token_cnt // np.mean(entity_size))
    np.random.shuffle(valid_entity)
    mask_idx = valid_entity[:min(mask_entity_cnt, valid_cnt), :]
    mask_pos = []
    for pos in mask_idx:
        mask_pos += list(range(pos[0], pos[1]))
    mask_pos = sorted(list(set(mask_pos)))
    return mask_pos

def apply_entity_mask_for_clm(src_len, src_entity, block_size=64, mask_prob=0.1, max_len=512):
    """
    apply continuous entity mask for Causal Language Model
    在entity周围mask，

    src_len = 256
    src_entity = [[2,4], [70,74], [175, 180]]

    return [2,3,4,5,6,7...20, 65,....90]
    """

    def get_side_idx(src_len, masked_pos):
        left_idxs = []
        right_idxs = []

        for idx in range(1, src_len - 1):
            if idx - 1 not in masked_pos and idx in masked_pos:
                left_idxs.append(idx)
            elif idx + 1 not in masked_pos and idx in masked_pos:
                right_idxs.append(idx)

        return left_idxs, right_idxs

    if len(src_entity) == 0:
        return []

    # 1. mask entity
    masked_pos = []
    for idx in range(len(src_entity)):
        ent_idx = list(range(src_entity[idx, 0], src_entity[idx, 1]))
        masked_pos.extend(ent_idx)

    # 2.  TODO: mask word around entity
    while len(masked_pos) < int(src_len * 0.3):
        left_idxs, right_idxs = get_side_idx(src_len, masked_pos)
        masked_idx_next = random.choice(left_idxs + right_idxs)
        if masked_idx_next in left_idxs:
            masked_pos.append(masked_idx_next - 1)
        else:
            masked_pos.append(masked_idx_next + 1)

    masked_pos = sorted(masked_pos)
    return masked_pos


def apply_random_mask(src_len, mask_prob=0.15, ignore_index=set()):
    """
    """
    candidates = [idx for idx in range(1, src_len - 1) if idx not in ignore_index]  # ignore bos and eos
    mask_token_cnt = math.ceil((src_len - len(ignore_index)) * mask_prob)
    random.shuffle(candidates)
    mask_pos = sorted(candidates[:mask_token_cnt])
    return mask_pos


def apply_span_mask(src_len, block_size=64, mask_prob=0.3):
    """ mask contiguous spans rather than random tokens
    - SpanBERT: mask contiguous span.
    - MASS:
    - BART: possioin
    - KPLUG: 30% clm_mask + 20% mlm_mask
    mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
    """
    positions = np.arange(0, src_len)
    masked_pos = []
    for i in range(0, src_len, block_size):
        block = positions[i: i + block_size]
        masked_len = int(len(block) * mask_prob)
        masked_block_start = np.random.choice(block[:len(block) - masked_len + 1], 1)[0]
        masked_pos.extend(positions[masked_block_start: masked_block_start + masked_len])
    return masked_pos


def apply_punc_deletion():
    """
    punc reconstruction: BART text_infilling
    motivation: for some noisy corpus with error punc.
    source: remove all punc
    """
    pass


def apply_gap_sentence_mask():
    """ ProphetNet """
    mask_pos = []
    return mask_pos


def apply_bart_mask():
    """ BART """
    pass


def apply_sentent_permutation():
    pass


def apply_entity_permutation():
    pass


def apply_ocr_segment_mask(src_item, mask_prob=0.1, ignore_index=None, max_len=512):
    pass
