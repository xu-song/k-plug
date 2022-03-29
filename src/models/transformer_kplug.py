# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fairseq import utils

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerEncoder,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from .transformer_mass import mass_base, mass_tiny, mass_big, TransformerMASSModel
from .transformer_hub_interface import TransformerHubInterface
from ..modules.output_heads import SentenceClassificationHead, MaskedLMHead, SequenceTaggingHead

logger = logging.getLogger(__name__)


@register_model('transformer_kplug')
class TransformerKplugModel(TransformerMASSModel):
    """
    K-PLUG: Knowledge-injected Pre-trained Language Model for Natural Language Understanding and Generation in E-Commerce.
    The pre-training procedure performs multi-task pretraining, including both NLU and NLG objective.
    """

    @classmethod
    def hub_models(cls):
        """
        TransformerForMaskedLM
        TransformerForCausalLM
        TransformerForSequenceClassification
        TransformerForTokenClassification
        TransformerForTokenClassification
        TransformerForQuestionAnswering
        TransformerForAbstractiveSummarization
        TransformerForExtractiveSummarization
        """

        # fmt: off
        def ner_config(path):
            return {
                'path': path,
                'tokenizer': None,
                'bpe': 'bert',
                'bpe_vocab_file': os.path.join(path, 'dict.txt'),
                'checkpoint_file': 'checkpoint72.20.pt',
            }

        def sum_jiadian_config(path):
            return {
                'path': path,
                'tokenizer': None,
                'bpe': 'bert',
                'bpe_vocab_file': os.path.join(path, 'dict.txt'), # hf 词典格式要单列
                'no_repeat_ngram_size': 3,
                'min_len': 50,
                'checkpoint_file': 'kplug_ft_jdsum_jiadian.pt',
                "task": "translation_bertdict",
            }

        def sum_as_lm_config(path):
            """ summarization model as language model """
            return {
                'path': path,
                'tokenizer': None,
                'bpe': 'bert',
                'bpe_vocab_file': os.path.join(path, 'dict.txt'),
                'no_repeat_ngram_size': 3,
                'min_len': 50,
                'checkpoint_file': 'checkpoint72.50.pt',
                'arch': 'transformer_kplug_lm_base',
                # below is optional args for outdated checkpoints
                "task": "masked_s2s",
                "criterion": "multitask_lm",
            }

        def pretrain_config(path):
            return {
                'path': path,
                'tokenizer': None,
                'bpe': 'bert',
                'bpe_vocab_file': os.path.join(path, 'dict.txt'),
                'no_repeat_ngram_size': 3,
                'min_len': 50,
                'checkpoint_file': 'checkpoint72.pt',
            }

        def pretrain_lm_config(path):
            """for language model tasks"""
            return {
                'path': path,
                'tokenizer': None,
                'bpe': 'bert',
                'bpe_vocab_file': os.path.join(path, 'dict.txt'),
                'no_repeat_ngram_size': 3,
                'min_len': 50,
                'checkpoint_file': 'checkpoint72.pt',
                'arch': 'transformer_kplug_lm_base',
                # below is optional args for outdated checkpoints
                "task": "masked_s2s_lm",
                "criterion": "multitask_lm",
            }

        def cls_config(path):
            pass

        HOME_PATH = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
        return {
            'transformer.pretrain': pretrain_config(HOME_PATH + '/models/fairseq/kplug/'),
            'kplug.pretrain.lm': pretrain_lm_config(HOME_PATH + '/models/fairseq/kplug/'),
            'transformer.ft.sum.lm': sum_as_lm_config(HOME_PATH + '/models/fairseq/kplug-finetune/sum/'),
            'transformer.ft.sum.jiadian': sum_jiadian_config(HOME_PATH + '/models/fairseq/kplug-finetune/jdsum/'),
            'transformer.ft.tag': ner_config(HOME_PATH + '/models/fairseq/kplug-finetune/ner_new/'),
            'transformer.ft.cls': cls_config(HOME_PATH + '/models/fairseq/kplug-finetune/cls/'),
        }

    @staticmethod
    def add_args(parser):
        TransformerMASSModel.add_args(parser)
        parser.add_argument('--share-encoder-input-output-embed', action='store_true',
                            help='for masked language model')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='checkpoint.pt', data_name_or_path='.', bpe='bert',
                        **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return TransformerHubInterface(x['args'], x['task'], x['models'])

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        model = super().build_model(args, task)
        model.mlm_head = MaskedLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(model.encoder.dictionary),
            activation_fn=args.activation_fn,
            weight=model.encoder.embed_tokens.weight if args.share_encoder_input_output_embed else None,
        )
        # or define in __init__ function
        model.classification_heads = nn.ModuleDict()
        model.tagging_heads = nn.ModuleDict()
        return model

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a sentence classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

        self.classification_heads[name] = SentenceClassificationHead(
            self.args.encoder_embed_dim if 'predicate_cls' not in name else self.args.encoder_embed_dim * 2,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    def register_tagging_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a sequence tagging head."""
        if name in self.tagging_heads:
            prev_num_classes = self.tagging_heads[name].out_proj.out_features
            prev_inner_dim = self.tagging_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.tagging_heads[name] = SequenceTaggingHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
            use_crf='crf' in name
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = TransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    def get_mlm_output(self, encoder_out, masked_tokens=None):
        """Project features to the vocabulary size."""
        return self.mlm_head(encoder_out, masked_tokens)

    def get_clm_output(self, decoder_out, masked_tokens=None):
        """sentence generation head, used for translation, summarization."""
        return decoder_out[masked_tokens]

    def get_cls_output(self, encoder_out, masked_tokens=None, classification_head_name=None,
                       src_subj_mask=None, src_obj_mask=None, src_ent_mask=None):
        """
        We "pool" the model by simply taking the hidden state corresponding to the first token.
        This is necessary for sentence-level classification tasks.
        input: [batch_size, seq_length, hidden_size]
        output: [batch_size, hidden_size]

        src_subj_mask & src_obj_mask is used for relation classification

        src_ent_mask is used for entity classification

        """
        if masked_tokens is not None:
            features = encoder_out[masked_tokens, :, :]
        else:
            features = encoder_out[:, :, :]

        if src_subj_mask is not None and src_obj_mask is not None:
            # For relation classification
            src_subj_mask = src_subj_mask[masked_tokens, :]
            src_obj_mask = src_obj_mask[masked_tokens, :]
            src_subj_len = src_subj_mask.sum(dim=1).unsqueeze(-1)
            src_obj_len = src_obj_mask.sum(dim=1).unsqueeze(-1)
            subj = torch.sum(features * src_subj_mask.unsqueeze(-1), dim=1) / src_subj_len  # B x C
            obj = torch.sum(features * src_obj_mask.unsqueeze(-1), dim=1) / src_obj_len  # B x C
            features = torch.cat([subj, obj], dim=-1)  # B x 2C
        elif src_ent_mask is not None:
            # For entity classification/typing
            src_ent_mask = src_ent_mask[masked_tokens, :]
            src_ent_len = src_ent_mask.sum(dim=1).unsqueeze(-1)
            features = torch.sum(features * src_ent_mask.unsqueeze(-1), dim=1) / src_ent_len  # B x C
        else:
            features = features[:, 0, :]  # B x C
        return self.classification_heads[classification_head_name](features)

    def get_tag_output(self, encoder_out, masked_tokens, tagging_head_name):
        return self.tagging_heads[tagging_head_name].decode(encoder_out, masked_tokens)

    def get_tag_loss(self, encoder_out, masked_tokens, tagging_head_name, tags):
        return self.tagging_heads[tagging_head_name](encoder_out, masked_tokens, tags)

    def slice_encoder_out(self, encoder_out, slice):
        new_out = {
            'encoder_out': [encoder_out['encoder_out'][0][:, slice, :]],
            'encoder_padding_mask': [encoder_out['encoder_padding_mask'][0][slice, :]],
            'encoder_embedding': [encoder_out['encoder_embedding'][0][slice, :, :]],
        }
        return new_out

    def forward(self, src_tokens=None, src_lengths=None, prev_output_tokens=None, prev_output_positions=None,
                masked_tokens=None, features_only=False, classification_head_name=None, tagging_head_name=None,
                tags=None):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            src_lengths (LongTensor): `(batch)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
        """
        if classification_head_name is not None and 'pretrain' not in classification_head_name:
            features_only = True

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        encoder_feature = encoder_out['encoder_out'][0].transpose(0, 1)  # T x B x C -> B x T x C

        # 1. encoder model
        if features_only:
            return encoder_feature

        # 2. encoder-decoder model
        decoder_out = None
        extra = {}
        if prev_output_tokens is not None and not prev_output_tokens.eq(self.decoder.padding_idx).all():
            if masked_tokens is not None:
                if (isinstance(masked_tokens, dict) and masked_tokens.get('decoder_mask', None) is not None):
                    encoder_out = self.slice_encoder_out(encoder_out, masked_tokens['decoder_mask'])

            decoder_out, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out,
                                              prev_output_positions=prev_output_positions)
            if torch.isnan(decoder_out).any():
                print('catch decoder nan')

        if masked_tokens is not None:
            if isinstance(masked_tokens, dict):
                if masked_tokens.get('clm', None) is not None:
                    extra['clm_out'] = self.get_clm_output(decoder_out, masked_tokens['clm'])
                if masked_tokens.get('mlm', None) is not None:
                    extra['mlm_out'] = self.get_mlm_output(encoder_feature, masked_tokens['mlm'])
                if masked_tokens.get('cls', None) is not None:
                    extra['cls_out'] = self.get_cls_output(encoder_feature, masked_tokens['cls'],
                                                           classification_head_name)  # masked_tokens['cls']是干嘛的？
                if masked_tokens.get('tag', None) is not None:
                    extra['tag_out'] = self.get_tag_loss(encoder_feature, masked_tokens['tag'], tagging_head_name, tags)
            elif isinstance(masked_tokens, torch.Tensor):
                decoder_out = self.get_clm_output(decoder_out, masked_tokens)

        return decoder_out, extra

    def get_mlm_targets(self, sample, net_output):
        return sample["mlm_target"]

    def get_clm_targets(self, sample, net_output):
        return sample["clm_target"]

    def get_cls_targets(self, sample, net_output):
        return sample["cls_target"]

    def get_tag_targets(self, sample, net_output):
        return sample["tag_target"]

    def get_multilabel_targets(self, sample, net_output):
        return sample["multilabel_target"]

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        prefix = name + '.' if name != '' else ''
        keys_to_delete = []

        # Handle new classification heads present in the state dict.
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)

            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                        num_classes != self.classification_heads[head_name].out_proj.out_features
                        or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)

        # Handle new tagging heads present in the state dict.
        current_tagging_head_names = [] if not hasattr(self, 'tagging_heads') else \
            list(self.tagging_heads.keys())
        for k in state_dict.keys():
            if not k.startswith(prefix + 'tagging_heads.'):
                continue

            head_name = k[len(prefix + 'tagging_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'tagging_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'tagging_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_tagging_head_names:
                    self.register_tagging_head(head_name, num_classes, inner_dim)

            else:
                if head_name not in current_tagging_head_names:
                    logger.warning(
                        'deleting tagging head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                        num_classes != self.tagging_heads[head_name].out_proj.out_features
                        or inner_dim != self.tagging_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting tagging head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)

        # delete ner_head in old version. git checkout aa117c57cf710309672c808b9973ea35633bf796
        for k in state_dict.keys():
            if k.startswith('ner_head.'):
                keys_to_delete.append(k)

        for k in keys_to_delete:
            del state_dict[k]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict['encoder.embed_tokens.weight'].size(0)
        if loaded_dict_size == len(self.encoder.dictionary) + 1 and '<mask>' not in self.encoder.dictionary:
            truncate_emb('encoder.embed_tokens.weight')
            truncate_emb('decoder.embed_tokens.weight')
            truncate_emb('encoder.output_projection.weight')
            truncate_emb('decoder.output_projection.weight')

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v

        if hasattr(self, 'tagging_heads'):
            cur_state = self.tagging_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'tagging_heads.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'tagging_heads.' + k)
                    state_dict[prefix + 'tagging_heads.' + k] = v

        # Copy mlm_head into state_dict
        cur_state = self.mlm_head.state_dict()
        for k, v in cur_state.items():
            k_str = prefix + 'mlm_head.' + k
            if k_str not in state_dict:
                logger.info('add ' + k_str + ' to loaded state_dict')
                state_dict[k_str] = v


@register_model('transformer_kplug_lm')
class TransformerLanguageModel(TransformerKplugModel):
    """ Auto-regressive language model

    ## Decoder LM  （需要把encoder藏起来）
    TransformerLanguageModel is a decoder only model.
    https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer_lm.py

    ## Encoder LM

    """

    def forward(self, src_tokens, **kwargs):
        """ forward decoder for recurrent language model """
        decoder_out = self.decoder(src_tokens, **kwargs)
        return decoder_out

    def forward_encoder(self, src_tokens, src_lengths=None, masked_tokens=None, **kwargs):
        """ forward encoder for masked language model """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        encoder_feature = encoder_out['encoder_out'][0].transpose(0, 1)
        mlm_out = self.get_mlm_output(encoder_feature, masked_tokens)
        return mlm_out


@register_model('transformer_mrc')
class TransformerMRCModel(TransformerKplugModel):
    """
    dataset：
      - zh：CMRC-2018 https://hfl-rc.github.io/cmrc2018/
                      https://github.com/DRCKnowledgeTeam/DRCD
      - en SQuAD、SQuADv2、 The Stanford Question Answering Dataset
        leaderboard：https://rajpurkar.github.io/SQuAD-explorer/
        <s> Passage here. </s> Q: Question here? </s>   https://github.com/ecchochan/roberta-squad
      - em NewsQA
      - em CommensenseQA
      - leaderboard：https://github.com/brightmart/roberta_zh#%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E6%B5%8B%E8%AF%95
    code：
      - https://github.com/CLUEbenchmark/CLUEPretrainedModels/tree/master/baselines/models_pytorch/mrc_pytorch
      - https://github.com/sogou/SogouMRCToolkit
      - https://github.com/zcgzcgzcg1/MRC_book
      - https://github.com/ewrfcas/bert_cn_finetune/blob/master/cmrc2018_finetune_pytorch.py
      - https://github.com/ewrfcas/bert_cn_finetune/blob/master/CJRC_finetune_pytorch.py
      - https://github.com/ewrfcas/bert_cn_finetune/blob/master/DRCD_finetune_pytorch.py

    """
    pass


@register_model('transformer_kplug_tagging')
class TransformerTaggingModel(TransformerKplugModel):
    """Bidirectional Language Model for Sequence Tagging"""

    def forward(self, src_tokens, masked_tokens=None, tagging_head_name=None, tags=None, **kwargs):
        encoder_out = self.encoder(src_tokens, **kwargs)
        log_likelihood = self.get_tag_loss(encoder_out['encoder_out'][0], masked_tokens, tagging_head_name, tags)
        return log_likelihood

    def decode(self, src_tokens, masked_tokens=None, tagging_head_name=None, **kwargs):
        encoder_out = self.encoder(src_tokens, **kwargs)
        # tags = self.get_tag_output(encoder_out['encoder_out'][0].transpose(0, 1), masked_tokens, tagging_head_name)
        tags = self.get_tag_output(encoder_out['encoder_out'][0], masked_tokens, tagging_head_name)
        return tags


@register_model('transformer_kplug_prediction')
class TransformerPredictionModel(TransformerKplugModel):
    """ Bidirectional Language Model for Sentence Prediction """

    def forward(self, src_tokens, features_only=True, classification_head_name=None, **kwargs):
        encoder_out = self.encoder(src_tokens, **kwargs)
        encoder_feature = encoder_out['encoder_out'][0].transpose(0, 1)  # T x B x C -> B x T x C
        cls_out = self.get_cls_output(encoder_feature, classification_head_name=classification_head_name)
        return cls_out, None


class TransformerForConditionalGeneration(TransformerKplugModel):
    """ endoder-decoder
    BartForConditionalGeneration https://github.com/huggingface/transformers/blob/155288f04ba9a5d0a0e4d5be4f6d4e808ad8cfff/src/transformers/modeling_bart.py#L940
    PegasusForConditionalGeneration https://github.com/huggingface/transformers/blob/155288f04ba9a5d0a0e4d5be4f6d4e808ad8cfff/src/transformers/modeling_pegasus.py#L24
    """
    pass


@register_model_architecture('transformer_kplug', 'transformer_kplug_base')
def transformer_kplug_base(args):
    args.share_encoder_input_output_embed = getattr(args, 'share_encoder_input_output_embed', True)
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    mass_base(args)


@register_model_architecture('transformer_kplug', 'transformer_kplug_tiny')
def transformer_tiny(args):
    args.share_encoder_input_output_embed = getattr(args, 'share_encoder_input_output_embed', True)
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    mass_tiny(args)

@register_model_architecture('transformer_kplug', 'transformer_kplug_big')
def transformer_tiny(args):
    args.share_encoder_input_output_embed = getattr(args, 'share_encoder_input_output_embed', True)
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    mass_big(args)



@register_model_architecture('transformer_kplug_tagging', 'transformer_kplug_tagging_base')
def transformer_tagging_base(args):
    transformer_kplug_base(args)


@register_model_architecture('transformer_kplug_prediction', 'transformer_kplug_prediction_base')
def transformer_prediction_base(args):
    transformer_kplug_base(args)


@register_model_architecture('transformer_kplug_lm', 'transformer_kplug_lm_base')
def transformer_tagging_base(args):
    transformer_kplug_base(args)
