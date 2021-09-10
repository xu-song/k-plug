# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
mass跟
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple


import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerModel,
    base_architecture,
)

from fairseq.modules.learned_positional_embedding import LearnedPositionalEmbedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.hub_utils import GeneratorHubInterface

logger = logging.getLogger(__name__)

class MASSDecoder(TransformerDecoder):
    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        prev_output_positions=None  # additional args
    ):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            prev_output_positions=prev_output_positions  # additional args
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            incremental_state=None,
            full_context_alignment=False,
            alignment_layer=None,
            alignment_heads=None,
            prev_output_positions=None,
    ):
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        if prev_output_positions is not None:
            positions = super(LearnedPositionalEmbedding, self.embed_positions).forward(prev_output_positions)
        else:
            positions = self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


@register_model('transformer_mass')
class TransformerMASSModel(TransformerModel):
    """
    1. --no-cross-attention  Cross+Self-Attention for Transformer Models
    2. Reducing Transformer Depth on Demand with Structured Dropout
    3. Quantization Noise for Extreme Model Compression
    """

    @classmethod
    def hub_models(cls):
        # fmt: off
        """
        https://github.com/pytorch/fairseq/tree/master/fairseq/data/encoders
        :return:
        """
        def space_bertbpe(path):
            return {
                'path': path,
                'tokenizer': 'space',
                'bpe': 'bert',
            }

        return {
            'mass.base': space_bertbpe('https://modelrelease.blob.core.windows.net/mass/mass-base-uncased.tar.gz'),
            'mass.middle': space_bertbpe('https://modelrelease.blob.core.windows.net/mass/mass-middle-uncased.tar.gz'),
            'mass.sum.base.cnndm': space_bertbpe('https://modelrelease.blob.core.windows.net/mass/cnndm_evaluation.tar.gz'),
            'mass.sum.base.gigaword': '',
            'mass.trans.base': '',
        }

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)

        # Arguments related to parameter initialization
        parser.add_argument('--apply-bert-init', action='store_true',
                            help='use custom param initialization for BERT')

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = TransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = MASSDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(self, src_tokens=None, src_lengths=None, prev_output_tokens=None, prev_output_positions=None, masked_tokens=None):
        """
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        x, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out, prev_output_positions=prev_output_positions)
        # for masked_lm criterion, https://github.com/pytorch/fairseq/blob/4f618a758ccd6b1924508ccbfb32eaacc3ea11c5/fairseq/criterions/masked_lm.py#L61
        if masked_tokens is not None:
            x = x[masked_tokens]
        return x, extra

    @staticmethod
    def upgrade_model_args(args):
        """Upgrade old model args for new versions args. For old
        """
        args.layernorm_embedding = True
        args.encoder_learned_pos = True
        args.decoder_learned_pos = True

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (old) state dict for new versions of fairseq.
        """
        mass_rename_keys = [
            ("encoder.emb_layer_norm.weight", "encoder.layernorm_embedding.weight"),
            ("encoder.emb_layer_norm.bias", "encoder.layernorm_embedding.bias"),
            ("decoder.emb_layer_norm.weight", "decoder.layernorm_embedding.weight"),
            ("decoder.emb_layer_norm.bias", "decoder.layernorm_embedding.bias"),
        ]
        def rename_key(state_dict, old, new):
            if old not in state_dict:
                return
            val = state_dict.pop(old)
            state_dict[new] = val
        for k, new_k in mass_rename_keys:
            rename_key(state_dict, k, new_k)

        super().upgrade_state_dict_named(state_dict, name)

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='checkpoint.pt', data_name_or_path='.', bpe='bert', **kwargs):
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
        return MASSHubInterface(x['args'], x['task'], x['models'])

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance. """
        cls.upgrade_model_args(args)
        model = super().build_model(args, task)
        return model


class ConditionalGeneration(TransformerMASSModel):
    pass


class MASSHubInterface(GeneratorHubInterface):
    pass


@register_model_architecture('transformer_mass', 'transformer_mass_base')
def mass_base(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)

    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    base_architecture(args)

""" other config:
- 3L-768H-12:
    - https://s3.amazonaws.com/models.huggingface.co/bert/clue/roberta_chinese_3L768_clue_tiny/config.json
    - https://github.com/CLUEbenchmark/CLUEPretrainedModels
- 
"""
@register_model_architecture('transformer_mass', 'transformer_mass_small')
def mass_small(args):
    """ """
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    mass_base(args)


@register_model_architecture('transformer_mass', 'transformer_mass_tiny')
def mass_tiny(args):
    """ """
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    mass_base(args)


@register_model_architecture('transformer_mass', 'transformer_mass_middle')
def mass_middle(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    mass_base(args)


@register_model_architecture('transformer_mass', 'transformer_mass_big')
def mass_big(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    mass_middle(args)
