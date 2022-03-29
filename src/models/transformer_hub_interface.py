# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import numpy as np
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import data_utils
from fairseq.hub_utils import GeneratorHubInterface
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints

logger = logging.getLogger(__name__)

class TransformerHubInterface(GeneratorHubInterface):

    def __init__(self, cfg, task, models):
        super().__init__(cfg, task, models)
        self.model = models[0]
        self.masked_token = '[MASK]'
        self.corrector = None


    def encode_masked_input(self, masked_input):
        text_spans = masked_input.split(self.masked_token)
        text_spans_bpe = (' {0} '.format(self.masked_token)).join(
            [self.bpe.encode(text_span.rstrip()) for text_span in text_spans]
        ).strip()
        tokens = self.task.source_dictionary.encode_line(
            '[CLS] ' + text_spans_bpe + ' [SEP]',
            append_eos=False,
            add_if_not_exist=False,
        )
        return tokens

    def encode_fn(self, x):
        x = self.tokenize(x)
        x = self.apply_bpe(x)
        return x

    def encode_constraints(self, constraints_list):
        """
        return tensor([[   2, 2841,    0,   16, 7398,    0]], device='cuda:0')
        """
        batch_constraints = [self.tgt_dict.encode_line(
            self.encode_fn(constraint),
            append_eos=False,
            add_if_not_exist=False,
        )
            for constraint in constraints_list]
        constraints_tensor = pack_constraints([batch_constraints])
        return constraints_tensor


    def encode_prefix_tokens(self, prefix_tokens):
        sentences = torch.stack([self.tgt_dict.encode_line(
            self.encode_fn(prefix_tokens),
            append_eos=False,
            add_if_not_exist=False,
        )]).long()
        if sentences[:, 0].eq(self.tgt_dict.bos()).all():
            sentences = sentences[:, 1:]
        return sentences


    def fill_single_mask(self, masked_inputs, topk=5):
        """
        :param masked_inputs:
        :param topk:
        :return:
        """
        masked_token = '[MASK]'
        assert all(masked_token in masked_input for masked_input in masked_inputs), \
            "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(masked_token)
        def encode_masked_input(masked_input):
            text_spans = masked_input.split(masked_token)
            text_spans_bpe = (' {0} '.format(masked_token)).join(
                [self.bpe.encode(text_span.rstrip()) for text_span in text_spans]
            ).strip()
            tokens = self.task.source_dictionary.encode_line(
                '[CLS] ' + text_spans_bpe + ' [SEP]',
                append_eos=False,
                add_if_not_exist=False,
            )
            return tokens

        tokens = [encode_masked_input(masked_input) for masked_input in masked_inputs]
        pad_to_length = max(len(token) for token in tokens)

        tokens = data_utils.collate_tokens(
            tokens,
            self.task.source_dictionary.pad(),
            self.task.source_dictionary.eos(),
            False, False,
            pad_to_length=pad_to_length,
        )
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        src_lengths = tokens.ne(self.task.source_dictionary.pad()).sum(dim=-1)
        masked_tokens = tokens.eq(self.task.source_dictionary.mask_index)


        with utils.model_eval(self.model):
            logits = self.model.forward_encoder(
                tokens.long().to(device=self.device),
                src_lengths=src_lengths.to(device=self.device),
                masked_tokens=masked_tokens
            )
        prob = logits.softmax(dim=-1)
        all_values, all_index = prob.topk(k=topk, dim=-1)
        # topk_predicted_token_bpe = self.task.source_dictionary.string(all_index)

        # topk_predicted_token_bpe = [tokens.split(' ') for tokens in topk_predicted_token_bpe.split('\n')]

        all_values = all_values.tolist()
        all_index = all_index.tolist()
        results = []
        for i in range(len(all_values)):
            masked_index = torch.nonzero(masked_tokens[i], as_tuple=False)
            result = []
            for v, p in zip(all_values[i], all_index[i]):
                filled_tokens = tokens[i].numpy()
                filled_tokens[masked_index] = p
                # Filter padding out:
                # tokens = tokens[np.where(tokens != self.task.source_dictionary.pad())]
                result.append(
                    {
                        "sequence": self.task.source_dictionary.string(filled_tokens),
                        "score": v,
                        "token": p,
                        "token_str": self.task.source_dictionary[p],
                    }
                )
            results.append(result)
        return result

    def fill_multi_mask(self, masked_inputs, topk=3, return_filled_sentence=False, how_select="argmax"):
        """ Fill one mask at a time, in L->R order
        Most of existing work support single mask.
        https://github.com/huggingface/transformers/blob/08f534d2da47875a4b7eb1c125cfa7f0f3b79642/src/transformers/pipelines.py#L1252
        """

        if how_select == "sample":
            pass
        elif how_select == "sample_topk":
            pass
        elif how_select == "argmax":
            pass
        else:
            raise NotImplementedError("Selection mechanism %s not found!" % how_select)
        pass

    def fill_mask(self, masked_inputs, topk=3, return_filled_sentence=False):
        """ TODO: batch support """
        if isinstance(masked_inputs, str):
            masked_inputs = [masked_inputs]
        # masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False)
        return self.fill_single_mask(masked_inputs)

    def disambiguate_pronoun(self, sentence: str) -> bool:
        """ Winograd Schema Challenge task (WSC)
        https://github.com/pytorch/fairseq/tree/master/examples/roberta#pronoun-disambiguation-winograd-schema-challenge
        """
        pass

    def register_classification_head(
            self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def register_tagging_head(
            self, name: str, num_tags: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_tagging_head(
            name, num_tags=num_tags, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        features = self.extract_features(tokens.to(device=self.device))
        logits = self.model.classification_heads[head](features)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def sequence_tagging(self, tokens: torch.LongTensor, head: str):
        tokens = tokens[:-1].view(1, -1)  # (batch, src_len)
        src_lengths = torch.LongTensor([tokens.numel()])
        tokens = utils.apply_to_sample(lambda t: t.to(self.device), tokens)  # TODO, build batch, 并加入到hub util
        masked_tokens = tokens.ne(self.src_dict.pad()).transpose(0, 1)  # (src_len, batch)
        tags = self.models[0].decode(tokens, tagging_head_name=head, masked_tokens=masked_tokens, src_lengths=src_lengths)



        hypo_tokens = np.array(
            tags[0]) + self.task.tag_dict.nspecial  # can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first
        hypo_str = self.task.tag_dict.string(hypo_tokens)

        src_tokens = utils.strip_pad(tokens, self.task.src_dict.pad())
        src_str = self.task.src_dict.string(src_tokens)
        print(src_str)
        print(hypo_str)
        return hypo_str

    def ppl(self, sentences, category=None):
        """ TODO: hierachical lm

        """
        if isinstance(sentences, str):
            sentences = [sentences]
        all_ppl = []
        scores = self.score(sentences)
        for score in scores:
            tokens = self.string(score['tokens']).split()  # tokenize and bpe
            probs = score['positional_scores'].exp()
            items = [{'word': token, 'prob': "%.6f" % float(prob)} for token, prob in zip(tokens, probs)]
            ppl = float(score['positional_scores'].mean().neg().exp())
            all_ppl.append({
                'ppl': ppl,
                'items': items,
            })
        return all_ppl

    def correct(self, text):
        if self.corrector is None:
            try:
                # from pycorrector.corrector import Corrector
                from ..correct_utils import BertCorrector as Corrector
                # from ..correct_utils import GPTCorrector as Corrector
                self.corrector = Corrector(self)
            except:
                logger.info('pip install pycorrector')
                return
        return self.corrector.correct(text)