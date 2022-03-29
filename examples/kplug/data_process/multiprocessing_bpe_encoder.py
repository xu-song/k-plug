"""
## 依赖项
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

## 运行
python bpe_encoder.py --vocab-bpe vocab.jd.txt  --inputs jd.train.raw --outputs jd.train.bpe  --workers 60

## 参考：
- https://github.com/microsoft/MASS/blob/master/MASS-summarization/encode.py
- https://github.com/pytorch/fairseq/blob/master/examples/roberta/multiprocessing_bpe_encoder.py


nohup python bpe_encoder.py \
    --vocab-bpe ../../vocab/vocab.jd.txt \
    --inputs raw/all.txt \
    --outputs bpe/all.bpe \
    --workers 60 > oov &

"""

import argparse
import contextlib
import sys
import os
import time

from collections import Counter
from multiprocessing import Pool

from transformers import BertTokenizer
from text_utils import TextProcessor

time_start = time.time()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help='path to vocab.bpe',
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            # if input != "-" else sys.stdin
            # 可能发生的错误：UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0
            if input != "-" else stack.enter_context(open(0, "r", encoding="utf-8", errors='ignore'))
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 1000)  # 这里并未计算, zip(* 是解压，
        # imap是map的延迟执行版本。
        # 给 chunksize 设置一个很大的值会比默认值 1 极大 地加快执行速度
        # 注意对于很长的迭代对象，可能消耗很多内存。可以考虑使用 imap() 或 imap_unordered() 并且显示指定 chunksize 以提升效率。

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):  # 主要的计算模块
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 100000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        print('total lines', i, file=sys.stderr)
        print('total time %.2f s' % (time.time() - time_start), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        # bpe = BertTokenizer.from_pretrained('bert-base-uncased')
        bpe = BertTokenizer(self.args.vocab_bpe)  # ------- 改动
        global p
        p = TextProcessor()


    def encode(self, line):
        global bpe
        global p
        line = p.clean(line)
        subword = bpe._tokenize(line)       # ---- 改动
        return subword                      # ---- 改动

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            # print('pid', os.getpid())
            if False and '[UNK]' in tokens:
                print(''.join(tokens))
                print(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
