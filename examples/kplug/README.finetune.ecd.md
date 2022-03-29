Finetune in E-commerce Dialogue Corpus
====

## Introduction

This is a task of binary classification. You can download raw dataset from https://github.com/cooelf/DeepUtteranceAggregation, 
or our binarized data from http://abc .

data format:
```
label \t conversation utterances (splited by \t) \t response
```

sampled data:
```
1	我 去 不 早 说 发韵 达 能 到 我家 那儿 我 就 能 拿到	韵达 不发 的 哦	那要 怎样 这 也 不行 那 也 不行	发 邮政 的 哦
0	我 去 不 早 说 发韵 达 能 到 我家 那儿 我 就 能 拿到	韵达 不发 的 哦	那要 怎样 这 也 不行 那 也 不行	那 不会 的 哦
0	我 去 不 早 说 发韵 达 能 到 我家 那儿 我 就 能 拿到	韵达 不发 的 哦	那要 怎样 这 也 不行 那 也 不行	韵达 邮政 EMS 随机 飞 随机 发 不 指定 快递 哦
```

## Finetune Pipeline

### 1) Data Preparation

```sh
DATA_DIR=data
mkdir ${DATA_DIR}/E-commerce-dataset/bpe

############# 1. text ############# 
# Tokenize source text
for split in train valid test; do
  python examples/kplug/data_process/multiprocessing_bpe_encoder.py \
    --vocab-bpe ${DATA_DIR}/vocab/vocab.txt \
    --inputs  ${DATA_DIR}/E-commerce-dataset/raw/${split}.txt \
    --outputs ${DATA_DIR}/E-commerce-dataset/bpe/${split}.bpe \
    --workers 60
done

# Binarized data
fairseq-preprocess \
    --user-dir src \
    --only-source \
    --task sentence_prediction_bertdict \
    --srcdict ${DATA_DIR}/vocab/vocab.txt \
    --trainpref ${DATA_DIR}/E-commerce-dataset/bpe/train.bpe \
    --validpref ${DATA_DIR}/E-commerce-dataset/bpe/valid.bpe  \
    --testpref ${DATA_DIR}/E-commerce-dataset/bpe/test.bpe  \
    --destdir ${DATA_DIR}/E-commerce-dataset/bin/input0  \
    --workers 60

############# 2. label ############# 
# construct label vocab (optional)
fairseq-preprocess \
  --only-source \
  --trainpref ${DATA_DIR}/E-commerce-dataset/raw/train.label \
  --destdir ${DATA_DIR}/E-commerce-dataset/all_label \
  --workers 60

# Binarized label
fairseq-preprocess \
    --only-source \
    --srcdict ${DATA_DIR}/E-commerce-dataset/all_label/dict.txt \
    --trainpref ${DATA_DIR}/E-commerce-dataset/raw/train.label  \
    --validpref ${DATA_DIR}/E-commerce-dataset/raw/valid.label  \
    --testpref ${DATA_DIR}/E-commerce-dataset/raw/test.label  \
    --destdir ${DATA_DIR}/E-commerce-dataset/bin/label  \
    --workers 60
```

### 2) Fine-tuning on ECD dataset:
Example fine-tuning 
```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
RESTORE_MODEL=models/fairseq/kplug/kplug.pt
DATA_DIR=data/E-commerce-dataset/bin/
HEAD_NAME=ecd_head      # Custom name for the classification head.
NUM_CLASSES=2           # Number of classes for the classification task.

fairseq-train $DATA_DIR \
    --user-dir src \
    --task sentence_prediction_bertdict \
    --arch transformer_kplug_prediction_base \
    --reset-optimizer --reset-dataloader --reset-meters \
    --criterion sentence_prediction \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 1e-05 --total-num-update 80000 --warmup-updates 4000 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --weight-decay 0.0 \
    --update-freq 8 \
    --max-tokens 20480 \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --max-sentences 64 \
    --ddp-backend=no_c10d --max-epoch 100 \
    --restore-file ${RESTORE_MODEL} \
    --fp16
```

It should be noted that parameter `--best-checkpoint-metric accuracy`


### 3) Inference for ECD test data using above trained checkpoint.

```
DATA_DIR=data/E-commerce-dataset/bin/
MODEL=models/fairseq/kplug-finetune/ecd/kplug_ft_ecd.pt  # 小模型

python fairseq_cli_ext/sentence_predict.py $DATA_DIR \
    --path $MODEL \
    --user-dir src \
    --task sentence_prediction_bertdict \
    --max-positions 512 \
    --classification-head-name ecd_head \
    --num-classes 2 \
    --batch-size 64 \
    2>&1 | tee output.txt


```

### 4) Evaluation


```
python eval_fairseq.py    
```




