Fine-tuning KPLUG on AVE dataset
====

## Introduction

This is a task of sequence tagging on AVE dataset.

Get full AVE dataset from https://github.com/jd-aig/JAVE.
And preprocess it into data files with non-tokenized cased samples.



### data sample


after BPE
```py
# train.bpe
不 用 确 认 了 ， 这 款 时     髦     复     古     的 个     性    原 创 夹       克       即 将 占 据 你 衣 橱 里 的 c 位 。

# train.tag
O  O  O  O  O  O  O  O  B-风格 I-风格 B-风格 I-风格 O  B-风格 I-风格 O O  B-产品词 I-产品词 O  O  O  O  O  O  O  O  O  O O O  
```


## Finetune Pipeline

### 1) BPE preprocess:

```
cd data_sample/ave/
python data_process.py
```

### 2) Binarize dataset:

you can binarize dataset by the following script or download binarized dataset from [](aa)

```
fairseq-preprocess \
  --user-dir src \
  --task sequence_tagging \
  --source-lang src \
  --target-lang tag \
  --srcdict data/vocab/vocab.jd.txt \
  --trainpref data/ave/bpe/train  \
  --validpref data/ave/bpe/valid \
  --testpref data/ave/bpe/test \
  --destdir data/ave/bin  \
  --workers 20
```



### 3) Fine-tuning on sequence tagging task:
Example fine-tuning 
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
#DATA_DIR=data
DATA_DIR=data_sample
DATA_BIN_DIR=${DATA_DIR}/ave/bin
RESTORE_MODEL=models/fairseq/kplug/kplug.pt
HEAD=ave_tagging_head

fairseq-train $DATA_BIN_DIR \
    --user-dir src \
    --task sequence_tagging \
    --arch transformer_kplug_tagging_base \
    --reset-optimizer --reset-dataloader --reset-meters \
    --criterion sequence_tagging \
    --source-lang src --target-lang tag \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 0.0005 --stop-min-lr 1e-09 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --weight-decay 0.0 \
    --update-freq 8 --max-tokens 4096 \
    --ddp-backend=no_c10d --max-epoch 50 \
    --max-source-positions 512 --max-target-positions 512 \
    --restore-file ${RESTORE_MODEL} \
    --tagging-head-name $HEAD --num-classes 53 \
    --fp16
```

if you need additional crf layer before prediction, just set `HEAD=crf_ave_tagging_head`


### 4) Inference for MEPAVE test data using above trained checkpoint.

```
export CUDA_VISIBLE_DEVICES=2
#DATA_DIR=data
DATA_DIR=data_sample
DATA_BIN_DIR=${DATA_DIR}/ave/bin
MODEL=models/fairseq/kplug-finetune/ave/kplug_ft_ave.pt
HEAD=ave_tagging_head

python fairseq_cli_ext/sequence_tagging.py $DATA_BIN_DIR \
    --path $MODEL \
    --user-dir src \
    --task sequence_tagging \
    --tagging-head-name $HEAD \
    --num-classes 53 \
    --source-lang src \
    --target-lang tag \
    --batch-size 64 \
    | tee output.txt

grep ^S output.txt | cut -f2- > src.txt
grep ^T output.txt | cut -f2- > tgt.txt
grep ^H output.txt | cut -f2- > hypo.txt

python examples/kplug/ner_utils/ner_acc.py src.txt hypo.txt tgt.txt
``` 



## Trouble Shooting

crf layer may not support multi gpu
```
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
        Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
https://github.com/pytorch/pytorch/issues/37377
```








