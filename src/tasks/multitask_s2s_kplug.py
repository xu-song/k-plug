# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import numpy as np
import logging
import torch

from collections import OrderedDict
from fairseq import utils
from fairseq.data import (
    data_utils,
    Dictionary,
    TruncateDataset,
    OffsetTokensDataset,
    StripTokenDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    TokenBlockDataset,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import register_task

from .multitask_s2s_mass import MaskedS2STask
from ..data.rstrip_token_dataset import RStripTokenDataset
from ..data.kplug_dataset import KnowledgeLanguagePairDataset

logger = logging.getLogger(__name__)


@register_task('multitask_lm')
class MultitaskLMTask(MaskedS2STask):
    """
    Train a multi-task sequence-to-sequence language model (mainly unsupervised tasks)
    """

    def __init__(self, args, data_dictionary, meta_dictionary):
        super().__init__(args, data_dictionary)
        self._meta_dictionary = meta_dictionary
        self.mask_idx = getattr(data_dictionary, 'mask_index', None) or data_dictionary.add_symbol("<mask>")

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        MaskedS2STask.add_args(parser)
        # parser.add_argument('--classification-head-name', type=str, default=None,
        #                     help='')  # has defined in criterion
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes or regression targets')
        parser.add_argument('--tagging-head-name', type=str, default=None,
                            help='')
        parser.add_argument('--tag-num-classes', type=int, default=-1,
                            help='number of tagging classes')
        parser.add_argument('--already-numberized', default=False, action='store_true',
                            help='already-numberized dataset')
        parser.add_argument('--sub-task',
                            type=str,
                            default='mlm_clm_segcls_titlegen',
                            help='activation function to use for pooler layer')

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        if args.classification_head_name:
            model.register_classification_head(
                args.classification_head_name,
                num_classes=self.cfg.num_classes,
            )
        if args.tagging_head_name:
            num_class = self.cfg.tag_num_classes if self.cfg.tag_num_classes > 0 else self.cfg.num_classes
            model.register_tagging_head(
                args.tagging_head_name,
                num_classes=num_class,
            )
        return model

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        data_dict = cls.load_dictionary(os.path.join(paths[0], 'input', 'dict.txt'))
        logger.info('dictionary: {} types'.format(len(data_dict)))
        meta_dict = Dictionary.load(os.path.join(paths[0], 'meta', 'dict.txt'))
        return cls(args, data_dict, meta_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """
        TODO:
          - break_mode="，。"
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        def get_path(type, split):
            return os.path.join(data_path, type, split)

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.cfg.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))
            return dataset

        dataset = make_dataset('input', self.dictionary)
        dataset = TruncateDataset(RStripTokenDataset(dataset, self.dictionary.eos()),
                                  self.cfg.tokens_per_sample - 2)
        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)。
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
        dataset = AppendTokenDataset(dataset, self.source_dictionary.eos())

        meta_dataset = make_dataset('meta', self.meta_dictionary)
        meta_dataset = StripTokenDataset(meta_dataset, id_to_strip=self.meta_dictionary.eos())
        s2s_dataset = KnowledgeLanguagePairDataset.apply_mask(
            dataset,
            dataset.sizes,
            self.source_dictionary,
            meta=meta_dataset,
            meta_sizes=meta_dataset.sizes,
            meta_dict=self.meta_dictionary,
            shuffle=True,
            mask_idx=self.mask_idx,
            mask_prob=self.cfg.mask_prob,
            leave_unmasked_prob=self.cfg.leave_unmasked_prob,
            random_token_prob=self.cfg.random_token_prob,
            sub_task=self.cfg.sub_task,
        )

        self.datasets[split] = s2s_dataset

    @property
    def meta_dictionary(self):
        return self._meta_dictionary
