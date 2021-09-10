

### 1) Data Preparation

```bash
cate=jiadian  # fushi  xiangbao
DATA_DIR=/workspace/fairseq/data
mkdir ${DATA_DIR}/sum/${cate}/bpe

# Tokenize
for split in train valid test; do
  for lang in src tgt; do
    python bpe_encoder.py \
      --vocab-bpe ${DATA_DIR}/vocab/vocab.txt \
      --inputs  ${DATA_DIR}/sum/sum/${cate}/raw/${split}.${lang} \
      --outputs ${DATA_DIR}/sum/sum/${cate}/bpe/${split}.${lang} \
      --workers 60
  done
done

# Binarized data
fairseq-preprocess \
  --user-dir /workspace/fairseq/src \
  --task translation \
  --bertdict \
  --source-lang src \
  --target-lang tgt \
  --srcdict ${DATA_DIR}/vocab/vocab.txt \
  --tgtdict ${DATA_DIR}/vocab/vocab.txt \
  --trainpref ${DATA_DIR}/sum/sum/${cate}/bpe/train  \
  --validpref ${DATA_DIR}/sum/sum/${cate}/bpe/valid \
  --testpref ${DATA_DIR}/sum/sum/${cate}/bpe/test \
  --destdir ${DATA_DIR}/sum/sum/${cate}/bin  \
  --workers 60
```


### 2) Fine-tuning on JD summarization task:

Example fine-tuning SUM
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_DIR=data/sum/jdsum/xiangbao/bin
RESTORE_MODEL=models/pretrain/checkpoint72.pt


nohup fairseq-train ${DATA_DIR} \
    --user-dir src \
    --task translation \
    --arch transformer_kplug_base \
    --bertdict \
    --reset-optimizer --reset-dataloader --reset-meters \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 0.0005 --stop-min-lr 1e-09 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --update-freq 8 --max-tokens 8192 \
    --ddp-backend=no_c10d --max-epoch 100 \
    --max-source-positions 512 --max-target-positions 512 \
    --truncate-source \
    --restore-file ${RESTORE_MODEL} \
    --fp16 > xiangbao1.20200926.log
```



### 3) Inference for JDSUM test data using above trained checkpoint.
```bash
export CUDA_VISIBLE_DEVICES=1
DATA_DIR=data/sum/jdsum/xiangbao/bin
MODEL=models/finetune/sum/checkpoint.jiadian.pt
MODEL=models/jd_finetune/sum/checkpoint72.21.jiadian.pt

fairseq-generate $DATA_DIR \
    --path $MODEL \
    --user-dir src \
    --task translation \
    --bertdict \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --batch-size 64 \
    --beam 5 \
    --min-len 50 \
    --truncate-source \
    --no-repeat-ngram-size 3 \
    2>&1 | tee output.txt

grep ^T output.txt | cut -f2- | sed 's/ ##//g' > tgt.txt
grep ^H output.txt | cut -f3- | sed 's/ ##//g' > hypo.txt
python get_rouge.py tgt.txt hypo.txt
```
