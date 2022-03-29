
CATEGORY=jiadian
MODEL_DIR=models/fairseq/kplug
MODEL=${MODEL_DIR}/kplug.pt

## 1. download pretrain model
if [ ! -f "$MODEL" ]; then
  mkdir -p $MODEL_DIR
  cd $MODEL_DIR
  wget http://storage.jd.com/language-models/kplug/pretrain/kplug.pt
  cd -
fi

## 2. Data Preparation
DATA_DIR=data_sample
mkdir ${DATA_DIR}/sum/cepsum/${CATEGORY}/bpe
echo "Applying BPE"
# Tokenize
for split in train valid test; do
  for lang in src tgt; do
    python examples/kplug/data_process/multiprocessing_bpe_encoder.py \
      --vocab-bpe ${DATA_DIR}/vocab/vocab.jd.txt \
      --inputs  ${DATA_DIR}/sum/cepsum/${CATEGORY}/raw/${split}.${lang} \
      --outputs ${DATA_DIR}/sum/cepsum/${CATEGORY}/bpe/${split}.${lang} \
      --workers 5
  done
done

# Binarized data
fairseq-preprocess \
  --user-dir src \
  --task translation_bertdict \
  --source-lang src \
  --target-lang tgt \
  --srcdict ${DATA_DIR}/vocab/vocab.jd.txt \
  --tgtdict ${DATA_DIR}/vocab/vocab.jd.txt \
  --trainpref ${DATA_DIR}/sum/cepsum/${CATEGORY}/bpe/train  \
  --validpref ${DATA_DIR}/sum/cepsum/${CATEGORY}/bpe/valid \
  --testpref ${DATA_DIR}/sum/cepsum/${CATEGORY}/bpe/test \
  --destdir ${DATA_DIR}/sum/cepsum/${CATEGORY}/bin  \
  --workers 5

## 3. Finetune
echo "Runing Finetune"
CATEGORY=jiadian
#DATA_DIR=data
DATA_DIR=data_sample
DATA_BIN_DIR=${DATA_DIR}/sum/cepsum/${CATEGORY}/bin
RESTORE_MODEL=models/fairseq/kplug/kplug.pt
MAX_EPOCH=2

fairseq-train ${DATA_BIN_DIR} \
    --user-dir src \
    --task translation_bertdict \
    --arch transformer_kplug_base \
    --source-lang src --target-lang tgt \
    --reset-optimizer --reset-dataloader --reset-meters \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 0.0005 --stop-min-lr 1e-09 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --update-freq 8 --max-tokens 8192 \
    --ddp-backend=no_c10d --max-epoch ${MAX_EPOCH} \
    --max-source-positions 512 --max-target-positions 512 \
    --truncate-source \
    --restore-file ${RESTORE_MODEL} \

## 4. Inference
echo "Runing Inference"
DATA_BIN_DIR=${DATA_DIR}/sum/cepsum/${CATEGORY}/bin
fairseq-generate $DATA_BIN_DIR \
    --path checkpoints/checkpoint_best.pt \
    --user-dir src \
    --task translation_bertdict \
    --batch-size 64 \
    --beam 5 \
    --min-len 50 \
    --truncate-source \
    --no-repeat-ngram-size 3 \
    2>&1 | tee ${CATEGORY}.output.txt


# 5. Evaluation






