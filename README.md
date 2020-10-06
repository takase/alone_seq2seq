# All Word Embeddings from One Embedding

This repository contains source files we used in our paper
>[All Word Embeddings from One Embedding](https://arxiv.org/abs/2004.12073)

>Sho Takase, Sosuke Kobayashi

## Requirements

- PyTorch version >= 1.4.0
- Python version >= 3.6
- For training new models, you'll also need an NVIDIA GPU and NCCL

## Machine Translation

We modified code to control output length, and thus the results might be slightly different from our paper.

Please use [released version](https://github.com/takase/alone_seq2seq/releases/tag/v1.0) to reproduce machine translation results in our paper.

### Training

##### 1. Download and pre-process datasets following the description in [this page](https://github.com/pytorch/fairseq/tree/master/examples/scaling_nmt)

##### 2. Train model

For binary mask with D_{inter} = 8192 using 4GPUs

```bash
python -u train.py \
    pre-processed-data-dir \
    --arch transformer_wmt_en_de --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 1.0 --lr 0.0015 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --warmup-init-lr 1e-07 --dropout 0.2 --weight-decay 0.0 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --max-tokens 3584 --min-lr 1e-09 --update-freq 32  --log-interval 100  --max-update 100000 \
    --one-emb binary --one-emb-relu-dropout 0.15 \
    --one-emb-layernum 2 --one-emb-inter-dim 8192  \
    --share-all-embeddings --stop-relu-dropout-update 4500 --save-dir model-save-dir
```

When you want to convert binary mask into real number filter, please set the following arguments:
```bash
    --one-emb real --one-emb-relu-dropout 0.2
```

### Test (decoding)

Averaging latest 10 checkpoints.

```bash
python scripts/average_checkpoints.py --inputs model-save-dir --num-epoch-checkpoints 10 --output model-save-dir/averaged.pt
```

Decoding with the averaged checkpoint.

```bash
python generate.py pre-processed-data-dir --path model-save-dir/averaged.pt  --beam 4 --lenpen 0.6 --remove-bpe | grep '^H' | sed 's/^H\-//g' | sort -t ' ' -k1,1 -n | cut -f 3-
```

## Summarization

In our paper, we used old pytorch and fairseq.

Please use [this code](https://github.com/takase/alone_old_seq2seq) to reproduce summarization results in our paper.

### Training

For binary mask with D_{inter} = 1024 using 4GPUs

```bash
python -u train.py \
    pre-processed-data-dir \
    --arch transformer_wmt_en_de --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 1.0 --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --warmup-init-lr 1e-07 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --max-tokens 3584 --min-lr 1e-09 --update-freq 16 --log-interval 100 --max-epoch 100 \
    --one-emb binary --one-emb-relu-dropout 0.1 \
    --one-emb-layernum 2 --one-emb-inter-dim 1024 \
    --share-all-embeddings --stop-relu-dropout-update 300 \
    --represent-length-by-lrpe --ordinary-sinpos --save-dir model-save-dir
```

### Test (decoding)

Averaging latest 10 checkpoints.

```bash
python scripts/average_checkpoints.py --inputs model-save-dir --num-epoch-checkpoints 10 --output model-save-dir/averaged.pt
```

Decoding with the averaged checkpoint.

```bash
python generate.py pre-processed-data-dir --path model-save-dir/averaged.pt  --beam 5 --desired-length 75
```

For comparison with the reported scores, use reranking following [this procedure](https://github.com/takase/control-length/tree/master/encdec).

## Acknowledgements

A large portion of this repo is borrowed from [fairseq](https://github.com/pytorch/fairseq).
