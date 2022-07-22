
CATEGORY=jiadian
MODEL=models/fairseq/kplug-finetune/cepsum/kplug_ft_cepsum_${CATEGORY}.pt
MODEL_DIR=models/fairseq/kplug-finetune/cepsum/

## 1. download model
if [ ! -f "$MODEL" ]; then
  mkdir -p $MODEL_DIR
  cd $MODEL_DIR
  echo "download model"
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

## 3. Inference
echo "Runing Inference"
DATA_BIN_DIR=${DATA_DIR}/sum/cepsum/${CATEGORY}/bin
fairseq-generate $DATA_BIN_DIR \
    --path $MODEL \
    --user-dir src \
    --task translation_bertdict \
    --batch-size 64 \
    --beam 5 \
    --min-len 50 \
    --truncate-source \
    --no-repeat-ngram-size 3 \
    2>&1 | tee ${CATEGORY}.output.txt




