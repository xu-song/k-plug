# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import torch

from collections import OrderedDict
from fairseq import utils
from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    RightPadDataset,
    OffsetTokensDataset,
    StripTokenDataset,
)
from fairseq.tasks import FairseqTask, register_task

from .multitask_s2s_mass import MaskedS2STask
from ..data.bert_dictionary import BertDictionary

logger = logging.getLogger(__name__)


@register_task('sequence_tagging')
class SequenceTaggingTask(FairseqTask):
    """
    encoder only model
    """

    def __init__(self, args, src_dict, tag_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tag_dict = tag_dict

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        MaskedS2STask.add_args(parser)
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes or regression targets')
        parser.add_argument('--tagging-head-name', type=str, default=None,
                            help='')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')

    @classmethod
    def setup_task(cls, args, **kwargs):
        """
        https://github.com/pytorch/fairseq/blob/master/fairseq/tasks/masked_lm.py#L78
        """
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))

        tag_dict = Dictionary.load(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tag_dict)))

        return cls(args, src_dict, tag_dict)

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        model.register_tagging_head(
            getattr(args, 'tagging_head_name', 'tagging_head'),
            num_classes=args.num_classes,
        )
        return model

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != getattr(self.cfg, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        prefix = os.path.join(data_path, '{}.{}-{}.'.format(split, src, tgt))
        src_dataset = data_utils.load_indexed_dataset(prefix + src, self.src_dict, self.cfg.dataset_impl)
        tag_dataset = data_utils.load_indexed_dataset(prefix + tgt, self.tag_dict, self.cfg.dataset_impl)

        src_dataset = StripTokenDataset(src_dataset, id_to_strip=self.source_dictionary.eos())
        tag_dataset = StripTokenDataset(tag_dataset, id_to_strip=self.tag_dictionary.eos())

        tag_pad = self.source_dictionary.pad()
        tag_offset = tag_pad + 1
        dataset = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': RightPadDataset(src_dataset, pad_idx=self.source_dictionary.pad()),
                'src_lengths': NumelDataset(src_dataset, reduce=False),
            },
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_dataset, reduce=True),
            'target': RightPadDataset(
                OffsetTokensDataset(tag_dataset, offset=-self.tag_dictionary.nspecial + tag_offset),
                pad_idx=tag_pad,
            ),
        }
        dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_dataset.sizes],
        )
        logger.info(str([self.src_dict[k] for k in dataset[0]['net_input.src_tokens']]))
        logger.info(str([self.tag_dict[k + self.tag_dictionary.nspecial - tag_offset] for k in
                         dataset[0]['target']]))
        self.datasets[split] = dataset

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.src_dict

    @property
    def tag_dictionary(self):
        return self.tag_dict

    def max_positions(self):
        max_positions = 1024
        if hasattr(self.cfg, 'max_positions'):
            max_positions = min(max_positions, self.cfg.max_positions)
        if hasattr(self.cfg, 'max_source_positions'):
            max_positions = min(max_positions, self.cfg.max_source_positions)
        if hasattr(self.cfg, 'max_target_positions'):
            max_positions = min(max_positions, self.cfg.max_target_positions)
        return (max_positions, max_positions)

    def inference_step(self, models, sample, tagging_head_name):
        with torch.no_grad():
            # masked_tokens = sample['target'].ne(self.source_dictionary.pad()).transpose(0, 1)
            masked_tokens = sample['net_input']['src_tokens'].ne(self.source_dictionary.pad()).transpose(0, 1)
            return models[0].decode(**sample['net_input'], masked_tokens=masked_tokens,
                                    tagging_head_name=tagging_head_name)
