

### 1) Data Preparation

```
DATA_DIR=/workspace/fairseq/data

# Tokenize
for split in train valid test; do
  for lang in src tgt; do
    python bpe_encoder.py \
      --vocab-bpe ${DATA_DIR}/vocab/vocab.txt \
      --inputs  ${DATA_DIR}/jddc/raw/${split}.${lang} \
      --outputs ${DATA_DIR}/jddc/bpe/${split}.${lang} \
      --workers 60
  done
done

fairseq-preprocess \
  --user-dir /workspace/fairseq/src \
  --task translation_mass \
  --source-lang src \
  --target-lang tgt \
  --srcdict ${DATA_DIR}/vocab/vocab.txt \
  --tgtdict ${DATA_DIR}/vocab/vocab.txt \
  --trainpref ${DATA_DIR}/jddc/bpe/train  \
  --validpref ${DATA_DIR}/jddc/bpe/valid \
  --testpref ${DATA_DIR}/jddc/bpe/test \
  --destdir ${DATA_DIR}/jddc/bin \
  --workers 20
```

### 2) Fine-tuning on JDDC task:
Example fine-tuning JDDC
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_DIR=data/jddc/bin
RESTORE_MODEL=models/pretrain/checkpoint.pt

nohup fairseq-train ${DATA_DIR} \
    --user-dir src \
    --task translation \
    --arch transformer_kplug_base \
    --bertdict \
    --reset-optimizer --reset-dataloader --reset-meters \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 0.0005 --min-lr 1e-09 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --update-freq 8 --max-tokens 4096 \
    --ddp-backend=no_c10d --max-epoch 100 \
    --max-source-positions 512 --max-target-positions 512 \
    --truncate-source \
    --restore-file ${RESTORE_MODEL} \
    --fp16
```



### 3) Inference for JDDC test data using above trained checkpoint.
```
export CUDA_VISIBLE_DEVICES=0
DATA_DIR=data/JD/jddc/bin
MODEL=models/finetune/jddc/checkpoint.pt

fairseq-generate $DATA_DIR \
    --path $MODEL \
    --user-dir src \
    --task translation_with_bertdict \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --batch-size 64 \
    --beam 5 \
    --truncate-source \
    --no-repeat-ngram-size 3 \
    2>&1 | tee output.txt

grep ^T output.txt | cut -f2- | sed 's/ ##//g' > tgt.txt
grep ^H output.txt | cut -f3- | sed 's/ ##//g' > hypo.txt
cat hypo.txt | sacrebleu tgt.txt
python get_rouge.py tgt.txt hypo.txt
```
