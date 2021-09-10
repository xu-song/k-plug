

### 1) Data Preparation

```
DATA_DIR=/workspace/fairseq/data
mkdir ${DATA_DIR}/qa_dialog/E-commerce-dataset/cls_task/bpe

# Tokenize
for split in train valid test; do
  python bpe_encoder.py \
    --vocab-bpe ${DATA_DIR}/vocab/vocab.txt \
    --inputs  ${DATA_DIR}/qa_dialog/E-commerce-dataset/cls_task/raw/${split}.txt \
    --outputs ${DATA_DIR}/qa_dialog/E-commerce-dataset/cls_task/bpe/${split}.bpe \
    --workers 60
done


# construct label vocab (optional)
#fairseq-preprocess \
#  --only-source \
#  --trainpref ${DATA_DIR}/qa_dialog/E-commerce-dataset/cls_task/raw/train.label \
#  --destdir ${DATA_DIR}/qa_dialog/E-commerce-dataset/cls_task/all_label \
#  --workers 60


# Binarized data
fairseq-preprocess \
    --user-dir /workspace/fairseq/src \
    --only-source \
    --task sentence_prediction \
    --srcdict ${DATA_DIR}/vocab/vocab.txt \
    --trainpref ${DATA_DIR}/qa_dialog/E-commerce-dataset/cls_task/bpe/train.bpe \
    --validpref ${DATA_DIR}/qa_dialog/E-commerce-dataset/cls_task/bpe/valid.bpe  \
    --testpref ${DATA_DIR}/qa_dialog/E-commerce-dataset/cls_task/bpe/test.bpe  \
    --destdir ${DATA_DIR}/qa_dialog/E-commerce-dataset/cls_task/bin/input0  \
    --workers 60

# label
fairseq-preprocess \
    --only-source \
    --srcdict ${DATA_DIR}/qa_dialog/E-commerce-dataset/cls_task/dict.label.txt \
    --trainpref ${DATA_DIR}/qa_dialog/E-commerce-dataset/cls_task/raw/train.label  \
    --validpref ${DATA_DIR}/qa_dialog/E-commerce-dataset/cls_task/raw/valid.label  \
    --testpref ${DATA_DIR}/qa_dialog/E-commerce-dataset/cls_task/raw/test.label  \
    --destdir ${DATA_DIR}/qa_dialog/E-commerce-dataset/cls_task/bin/label  \
    --workers 60
```

### 2) Fine-tuning on ECD dataset:
Example fine-tuning 
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
RESTORE_MODEL=models/pretrain/checkpoint.pt
DATA_DIR=data/qa_dialog/E-commerce-dataset/cls_task/bin/

fairseq-train $DATA_DIR \
    --user-dir src \
    --task sentence_prediction \
    --arch transformer_prediction_base \
    --reset-optimizer --reset-dataloader --reset-meters \
    --criterion sentence_prediction \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 1e-05 --warmup-updates 4000 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --weight-decay 0.0 \
    --update-freq 8 \
    --max-tokens 20480 \
    --max-positions 512 \
    --max-sentences 64 \
    --ddp-backend=no_c10d --max-epoch 100 \
    --restore-file ${RESTORE_MODEL} \
    --classification-head-name catecls_head \
    --num-classes 2 \
    --fp16
```

It should be noted that parameter `--best-checkpoint-metric accuracy`


### 3) Inference for ECD test data using above trained checkpoint.

```
cd fairseq_cli_ext/

export CUDA_VISIBLE_DEVICES=0
DATA_DIR=../data/qa_dialog/E-commerce-dataset/cls_task/bin/
MODEL=../models/finetune/ali_cls/checkpoint.pt

python sentence_predict.py $DATA_DIR \
    --path $MODEL \
    --user-dir ../src \
    --task bert_sentence_prediction \
    --max-positions 512 \
    --classification-head-name catecls_head \
    --num-classes 2 \
    --batch-size 64 \
    2>&1 | tee output.txt

python eval_fairseq.py    
```

