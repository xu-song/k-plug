# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch

from fairseq.data import FairseqDataset


class MultiLabelDataset(FairseqDataset):
    """convert categorical labels to multi-hot labels."""

    def __init__(self, labels, num_labels):
        super().__init__()
        self.labels = labels
        self.num_labels = num_labels

    def __getitem__(self, index):
        multi_hot_label = torch.zeros(self.num_labels)
        multi_hot_label[self.labels[index]] = 1
        return multi_hot_label

    def __len__(self):
        return len(self.labels)
