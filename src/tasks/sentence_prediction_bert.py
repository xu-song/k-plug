# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.data import Dictionary
from fairseq.tasks import register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask
from ..data.bert_dictionary import BertDictionary


@register_task('bert_sentence_prediction')
class BertSentencePredictionTask(SentencePredictionTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.
    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        SentencePredictionTask.add_args(parser)
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--max-source-positions', default=512, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=512, type=int, metavar='N',
                            help='max number of tokens in the target sequence')

    @classmethod
    def load_dictionary(cls, args, filename=None, source=True):
        if filename == None:
            filename = args
        if 'label' in filename:
            return Dictionary.load(filename)
        return BertDictionary.load_from_file(filename)

    def inference_step(self, models, sample, classification_head_name):
        with torch.no_grad():
            logits, _ = models[0](
                **sample['net_input'],
                features_only=True,
                classification_head_name=classification_head_name,
            )
            probs = torch.softmax(logits, dim=1)
            pred_scores, pred_labels = torch.max(probs, 1)
            return {
                'labels': pred_labels,
                'scores': pred_scores,
            }
