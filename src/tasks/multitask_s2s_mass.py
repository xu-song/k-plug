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
    PrependTokenDataset,
    LanguagePairDataset,
    TokenBlockDataset,
    AppendTokenDataset,
    data_utils,
    Dictionary,
    IdDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    TransformEosDataset,
    TruncatedDictionary,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks.language_modeling import LanguageModelingTask

from ..data.mask_language_pair_dataset import MaskedLanguagePairDataset
from ..data.bert_dictionary import BertDictionary

logger = logging.getLogger(__name__)


@register_task('masked_s2s')
class MaskedS2STask(FairseqTask):
    """
    Train a masked sequence-to-sequence model.
    MASS: Masked Sequence to Sequence Pre-training for Language Generation, ICML 2019

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--sample-break-mode', default='none',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments '
                                 'per sample for BERT dataset')
        # parser.add_argument('--max-source-positions', default=512, type=int, metavar='N',
        #                     help='max number of tokens in the source sequence')
        # parser.add_argument('--max-target-positions', default=512, type=int, metavar='N',
        #                     help='max number of tokens in the target sequence')
        parser.add_argument('--mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
                            help='probability that a masked token is unmasked')
        parser.add_argument('--random-token-prob', default=0.1, type=float,
                            help='probability of replacing a token with a random token')
        parser.add_argument('--freq-weighted-replacement', default=False, action='store_true',
                            help='sample random replacement words based on word frequencies')
        parser.add_argument('--mask-whole-words', default=False, action='store_true',
                            help='mask whole words; you may also want to set --bpe')
        parser.add_argument('--shorten-method', default='none',
                            choices=['none', 'truncate', 'random_crop'],
                            help='if not none, shorten sequences that exceed --tokens-per-sample')
        parser.add_argument('--shorten-data-split-list', default='',
                            help='comma-separated list of dataset splits to apply shortening to, '
                                 'e.g., "train,valid" (default: all dataset splits)')
        parser.add_argument('--shuffle', action='store_true',
                            help='shuffle each dataset while training')

        # fmt: on

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """
        https://github.com/pytorch/fairseq/blob/master/fairseq/tasks/masked_lm.py#L78
        """
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        dictionary = cls.load_dictionary(os.path.join(paths[0], 'dict.txt'))
        logger.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.dictionary,
            self.cfg.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,
        )

        # create continuous blocks of tokens.
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.cfg.tokens_per_sample,
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.cfg.sample_break_mode,
        )
        logger.info('loaded {} blocks from: {}'.format(len(dataset), split_path))
        s2s_dataset = MaskedLanguagePairDataset.apply_mask(
            dataset,
            dataset.sizes,
            self.source_dictionary,
            shuffle=True,
            mask_prob=self.cfg.mask_prob,
            leave_unmasked_prob=self.cfg.leave_unmasked_prob,
            random_token_prob=self.cfg.random_token_prob,
        )
        self.datasets[split] = s2s_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def max_positions(self):
        max_positions = 1024
        if hasattr(self.cfg, 'max_positions'):
            max_positions = min(max_positions, self.cfg.max_positions)
        if hasattr(self.cfg, 'max_source_positions'):
            max_positions = min(max_positions, self.cfg.max_source_positions)
        if hasattr(self.cfg, 'max_target_positions'):
            max_positions = min(max_positions, self.cfg.max_target_positions)
        return (max_positions, max_positions)

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = PrependTokenDataset(
            dataset,
            token=(
                self.source_dictionary.bos()
                if getattr(self.cfg, "add_bos_token", False)
                else self.source_dictionary.eos()
            ),
        )
        tgt_dataset = AppendTokenDataset(
            dataset,
            token=self.source_dictionary.pad()
        )
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(src_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False),
            },
            sizes=[np.array(src_lengths)],
        )



@register_task('masked_s2s_lm')
class MASSLMTask(MaskedS2STask):
    """ multiple inheritance: MaskedS2STask  LanguageModelingTask
    TODO: 兼容encoder和decoder
    """

    def inference_step_by_decoder(self, generator, models, sample, prefix_tokens=None):
        """
        call forward_decoder(
           https://github.com/pytorch/fairseq/blob/3c414780837dd3506ea82a868ea92628d1fdd576/fairseq/models/fairseq_model.py#L501
        """
        raise NotImplementedError

    def inference_step_by_encoder(self, generator, models, sample, prefix_tokens=None):
        """
        call forward_encoder(
        """
        raise NotImplementedError

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        """
        decoder based language model.
        Reference: fairseq/tasks/language_modeling.py
        For: score、sample、generate, not used for fill_mask
        """
        with torch.no_grad():
            # remove encoder temporarily
            # encoder = models[0].encoder
            # delattr(models[0], 'encoder')

            # Generation will always be conditioned on bos_token
            if getattr(self.cfg, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            if constraints is not None:
                raise NotImplementedError("Constrained decoding with the language_modeling task is not supported")

            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                if prefix_tokens[:, 0].eq(bos_token).all():
                    prefix_tokens = prefix_tokens[:, 1:]

            result = generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token,
            )
            # models[0].encoder = encoder
            return result

