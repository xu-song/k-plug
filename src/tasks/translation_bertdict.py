# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, TranslationConfig
from ..data.bert_dictionary import BertDictionary


@register_task("translation_bertdict", dataclass=TranslationConfig)
class TranslationBertdictTask(TranslationTask):

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)