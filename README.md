# K-PLUG

[K-PLUG: Knowledge-injected Pre-trained Language Model for Natural Language Understanding and Generation in E-Commerce](https://aclanthology.org/2021.findings-emnlp.1/)
(Findings of EMNLP 2021),
by Song Xu, Haoran Li, Peng Yuan, Yujia Wang, Youzheng Wu, Xiaodong He, Ying Liu, and Bowen Zhou, 
is a knowledge-injected pre-trained language model based on the encoder-decoder framework that can be transferred to both natural language understanding and generation tasks.


## What's New:
- **March 2022** Released [M-KPLUG](https://github.com/WaveLi123/m-kplug) which injects the visual signals to the decoder layer.
- **April 2022** Released [demo](finetune_cepsum_demo.sh) for Shared Tasks in NLPCC 2022 [Multimodal Summarization Challenge](https://jd-nlg-rhino.github.io/)

## Quick Start with Docker (fairseq)


GPU
```sh
git clone https://github.com/xu-song/k-plug.git
cd k-plug

nvidia-docker run -it --shm-size 15G --network=host -v $(pwd):/workspace/ bitspeech/fairseq:latest bash
sh finetune_cepsum_demo.sh  # 1. download model 2. finetune 3. inference 4. evaluation
```

CPU
```sh
git clone https://github.com/xu-song/k-plug.git
cd k-plug

docker run -it --network=host -v $(pwd):/workspace/ bitspeech/fairseq:latest bash
sh finetune_cepsum_demo.sh
```

## Quick Start with HuggingFace

We also provide pretrained model in huggingface version.

For more details, please refer to [huggingface demo](examples/kplug_hf)


## Model Zoo

We provide kplug pretrain and finetune models. 

Encoder: 6L-768H-12A, Decoder: 6L-768H-12A, 110M parameters.


| Task | Task Type | Task Description |  Model |
|---|---|---|---|
| pre-train | NLU & NLG | multi-task pre-train | [kplug.pt](http://storage.jd.com/language-models/kplug/pretrain/kplug.pt)  |
| ft-MEPAVE | NLU | sequence tagging | [kplug_ft_ave.pt](http://storage.jd.com/language-models/kplug/ft-ave/kplug_ft_ave.pt) |
| ft-ECD | NLU | retrieval based chatbot | [kplug_ft_ecd.pt](http://storage.jd.com/language-models/kplug/ft-ecd/kplug_ft_ecd.pt) |
| ft-JDDC | NLG | generation based chatbot |  [kplug_ft_jddc.pt](http://storage.jd.com/language-models/kplug/ft-jddc/kplug_ft_jddc.pt) |







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

[Finetuning on CEPSUM Dataset (Abstractive Summarization)](examples/kplug/README.finetune.summarization.cepsum.md)

[Finetuning on MEPAVE Dataset (Sequence Tagging)](examples/kplug/README.finetune.seq_tagging.md)





## Dependencies

Currently we implement K-PLUG based on fairseq. The dependencies are as follows:

- PyTorch version >= 1.5.0
- Python version >= 3.6    
- fairseq 
```sh
git clone https://github.com/pytorch/fairseq.git
cd fairseq 
pip install --editable ./
# python setup.py build_ext --inplace
```



## Reference

If you use this code please cite our paper:
```
@inproceedings{xu-etal-2021-k-plug,
    title = "K-{PLUG}: Knowledge-injected Pre-trained Language Model for Natural Language Understanding and Generation in {E}-Commerce",
    author = {Xu, Song and Li, Haoran and Yuan, Peng and Wang, Yujia and Wu, Youzheng and He, Xiaodong and Liu, Ying and Zhou, Bowen},
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.1"
}
```
