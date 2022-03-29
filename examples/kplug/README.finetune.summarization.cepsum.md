Finetuning on CEPSUM Dataset 
====

## Introduction

This is a task of abstractive summarization on [CEPSUM dataset](https://github.com/hrlinlp/cepsum),
which contains 1.4 million instances collected from a major Chinese
e-commerce platform, covering three categories of product: Home Appliances, Clothing, and Cases & Bags.


### data sample

test.src
```
美的对开门风冷无霜家用智能电冰箱波光金纤薄机身高颜值助力保鲜，美的家居风，尺寸说明：M以上的距离尤其是左右两侧距离必须保证。关于尺寸的更多问题可，LED冷光源，纤薄机身，风冷无霜，智能操控，远程调温，节能静音，照亮你的视野，535L大容量，系统散热和使用的便利性，建议左右两侧、顶部和背部需要预留10C，电源线和调平脚等。冰箱放置时为保证，菜谱推荐，半开门俯视图，全开门俯视图，预留参考图
```

test.tgt
```
独立双循环制冷，全方位风冷，缔造无霜时代，食物保鲜更长久，取放食物更自由。610L超大空间，满足一家人的储鲜需求不是问题。高效变频压缩机，节省的是能源，改变的是时代，创造的是生活。
```

generate sample
```
S-0 美 的 对 开 门 风 冷 无 霜 家 用 智 能 电 冰 箱 波 光 金 纤 薄 机 身 高 颜 值 助 力 保 鲜 ， 美 的 家 居 风 ， 尺 寸 说 明 ： m 以 上 的 距 离 尤 其 是 左 右 两 侧 距 离 必 须 保 证 。 关 于 尺 寸 的 更 多 问 题 可 ， led 冷 光 源 ， 纤 薄 机 身 ， 风 冷 无 霜 ， 智 能 操 控 ， 远 程 调 温 ， 节 能 静 音 ， 照 亮 你 的 视 野 ， 53 ##5 ##l 大 容 量 ， 系 统 散 热 和 使 用 的 便 利 性 ， 建 议 左 右 两 侧 、 顶 部 和 背 部 需 要 预 留 10 ##c ， 电 源 线 和 调 平 脚 等 。 冰 箱 放 置 时 为 保 证 ， 菜 谱 推 荐 ， 半 开 门 俯 视 图 ， 全 开 门 俯 视 图 ， 预 留 参 考 图
T-0 独 立 双 循 环 制 冷 ， 全 方 位 风 冷 ， 缔 造 无 霜 时 代 ， 食 物 保 鲜 更 长 久 ， 取 放 食 物 更 自 由 。 61 ##0 ##l 超 大 空 间 ， 满 足 一 家 人 的 储 鲜 需 求 不 是 问 题 。 高 效 变 频 压 缩 机 ， 节 省 的 是 能 源 ， 改 变 的 是 时 代 ， 创 造 的 是 生 活 。
H-0 -0.3718273937702179 这 款 美 的 冰 箱 ， 采 用 优 质 的 变 频 电 机 ， 稳 定 的 性 能 有 效 减 少 噪 音 的 影 响 ， 利 用 风 冷 无 霜 的 制 冷 技 术 ， 能 够 很 好 保 留 食 物 的 水 分 ， 贴 心 呵 护 家 人 的 食 物 健 康 ， 精 致 大 气 的 外 观 设 计 ， 尽 显 出 时 尚 的 厨 房 魅 力 。
```




## Finetune Pipeline

### 1) Data Preparation

```bash
CATEGORY=jiadian  # fushi xiangbao
#DATA_DIR=data
DATA_DIR=data_sample
mkdir ${DATA_DIR}/sum/cepsum/${CATEGORY}/bpe

# Tokenize
for split in train valid test; do
  for lang in src tgt; do
    python examples/kplug/data_process/multiprocessing_bpe_encoder.py \
      --vocab-bpe ${DATA_DIR}/vocab/vocab.jd.txt \
      --inputs  ${DATA_DIR}/sum/cepsum/${CATEGORY}/raw/${split}.${lang} \
      --outputs ${DATA_DIR}/sum/cepsum/${CATEGORY}/bpe/${split}.${lang} \
      --workers 60
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
  --workers 60
```


### 2) Fine-tuning on JD summarization task:

Example fine-tuning SUM
```bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3

CATEGORY=jiadian  # fushi xiangbao
#DATA_DIR=data
DATA_DIR=data_sample
DATA_BIN_DIR=${DATA_DIR}/sum/cepsum/${CATEGORY}/bin
RESTORE_MODEL=models/fairseq/kplug/kplug.pt


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
    --ddp-backend=no_c10d --max-epoch 100 \
    --max-source-positions 512 --max-target-positions 512 \
    --truncate-source \
    --restore-file ${RESTORE_MODEL} \
    --fp16
```



### 3) Inference for CEPSUM test data using above trained checkpoint.
```bash
# 家电、服饰、箱包
CATEGORY=jiadian  # fushi xiangbao
#DATA_DIR=data
DATA_DIR=data_sample
DATA_BIN_DIR=${DATA_DIR}/sum/cepsum/${CATEGORY}/bin # 
MODEL=models/fairseq/kplug-finetune/cepsum/kplug_ft_cepsum_${CATEGORY}.pt

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


```


## 4) Evaluation

[files2rouge](https://github.com/pltrdy/files2rouge) is required to evaluate kplug on summarization task.

```
grep ^T ${CATEGORY}.output.txt | cut -f2- | sed 's/ ##//g' > tgt.txt
grep ^H ${CATEGORY}.output.txt | cut -f3- | sed 's/ ##//g' > hypo.txt
python tools/get_rouge.py tgt.txt hypo.txt
```
