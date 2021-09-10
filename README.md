## Introduction

K-PLUG: Knowledge-injected Pre-trained Language Model for Natural Language Understanding and Generation in E-Commerce.

## Installation

- PyTorch version >= 1.5.0
- Python version >= 3.6    

```
git clone https://github.com/pytorch/fairseq.git
cd fairseq 
pip install --editable ./
```

## Pre-training

prepare data for pre-training [train.sh](data_process/prepare_pretrain.sh)

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3

function join_by { local IFS="$1"; shift; echo "$*"; }
DATA_DIR=$(join_by : data/kplug/bin/part*)

USER_DIR=src
TOKENS_PER_SAMPLE=512
WARMUP_UPDATES=10000
PEAK_LR=0.0005
TOTAL_UPDATES=125000
#MAX_SENTENCES=8
MAX_SENTENCES=16
UPDATE_FREQ=16   # batch_size=update_freq*max_sentences*nGPU = 16*16*4 = 1024

SUB_TASK=mlm_clm_sentcls_segcls_titlegen 
## ablation task
#SUB_TASK=clm_sentcls_segcls_titlegen
#SUB_TASK=mlm_sentcls_segcls_titlegen
#SUB_TASK=mlm_clm_sentcls_segcls
#SUB_TASK=mlm_clm_segcls_titlegen
#SUB_TASK=mlm_clm_sentcls_titlegen

fairseq-train $DATA_DIR \
    --user-dir $USER_DIR \
    --task multitask_lm \
    --sub-task $SUB_TASK \
    --arch transformer_pretrain_base \
    --min-loss-scale=0.000001 \
    --sample-break-mode none \
    --tokens-per-sample $TOKENS_PER_SAMPLE \
    --criterion multitask_lm \
    --apply-bert-init \
    --max-source-positions 512 --max-target-positions 512 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR \
    --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --ddp-backend=no_c10d \
    --tensorboard-logdir tensorboard \
    --classification-head-name pretrain_head --num-classes 40 \
    --tagging-head-name pretrain_tag_head --tag-num-classes 2 \
    --fp16
```

## Fine-tuning and Inference

[Finetuning on JDDC (Response Generation)](examples/kplug/README.finetune.jddc.md)

[Finetuning on ECD Corpus (Response Retrieval)](examples/kplug/README.finetune.ecd.md)

[Finetuning on JD Product Dataset (Abstractive Summarization)](examples/kplug/README.finetune.summarization.cate.md)

[Finetuning on MEPAVE Dataset (Sequence Tagging)](examples/kplug/README.finetune.seq_tagging.md)

