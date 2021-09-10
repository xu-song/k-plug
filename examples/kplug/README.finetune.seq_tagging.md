# Fine-tuning KPLUG on AVE task

Download the AVE data and preprocess it into data files with non-tokenized cased samples.

### 1) BPE preprocess:


### 2) Binarize dataset:

```
DATA_DIR=/workspace/fairseq-models/data
# Binarized data 
fairseq-preprocess \
  --user-dir /workspace/fairseq/src \
  --task sequence_tagging \
  --source-lang src \
  --target-lang tag \
  --srcdict ${DATA_DIR}/vocab/vocab.txt \
  --trainpref ${DATA_DIR}/kb_ner/bpe/train  \
  --validpref ${DATA_DIR}/kb_ner/bpe/valid \
  --testpref ${DATA_DIR}/kb_ner/bpe/test \
  --destdir ${DATA_DIR}/kb_ner/bin  \
  --workers 20
```



### 3) Fine-tuning on sequence tagging task:
Example fine-tuning 
```
export CUDA_VISIBLE_DEVICES=0,1,2,3

RESTORE_MODEL=models/transformer_pretrain/checkpoint.pt
HEAD=jd_tagging_head

fairseq-train data/JD/kb_ner/bin \
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



### 4) Inference for MEPAVE test data using above trained checkpoint.

```
cd fairseq_cli_ext/

export CUDA_VISIBLE_DEVICES=2
DATA_DIR=../data/JD/kb_ner/bin
MODEL=../models/jd_finetune/ner_new/checkpoint72.20.pt
HEAD=transformer_tagging_head

python sequence_predict.py $DATA_DIR \
    --path $MODEL \
    --user-dir ../src \
    --task sequence_tagging \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --tagging-head-name $HEAD \
    --num-classes 53 \
    --source-lang src \
    --target-lang tag \
    --batch-size 64 \
    2>&1 | tee output.txt

grep ^S output.txt | cut -f2- > src.txt
grep ^T output.txt | cut -f2- > tgt.txt
grep ^H output.txt | cut -f2- > hypo.txt

data_dir=/workspace/fairseq/examples/pretrain/ner_utils/
mkdir $data_dir
mv src.txt $data_dir
mv tgt.txt $data_dir
mv hypo.txt $data_dir

cd ../examples/pretrain/ner_utils/
python convert_bio_to_char.py
python ner_acc.py
```

if you need additional crf layer before prediction, just set `HEAD=crf_tagging_head`


## Trouble Shooting

crf layer may not support multi gpu
```
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
        Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
https://github.com/pytorch/pytorch/issues/37377
```








