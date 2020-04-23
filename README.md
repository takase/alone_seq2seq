# All Word Embeddings from One Embedding

## Requirements

- PyTorch version >= 1.4.0
- Python version >= 3.6
- For training new models, you'll also need an NVIDIA GPU and NCCL

## Machine Translation

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
    --one-emb real --one-emb-relu-dropout 0.15
```

##### 3. Test (decoding)

Averaging latest 10 checkpoints.

```bash
python scripts/average_checkpoints.py --inputs model-save-dir --num-epoch-checkpoints 10 --output model-save-dir/averaged.pt
```

Decoding with the averaged checkpoint.

```bash
python generate.py pre-processed-data-dir --path model-save-dir/averaged.pt  --beam 4 --lenpen 0.6 --remove-bpe | grep '^H' | sed 's/^H\-//g' | sort -t ' ' -k1,1 -n | cut -f 3-
```

## Acknowledgements

A large portion of this repo is borrowed from [fairseq](https://github.com/pytorch/fairseq).