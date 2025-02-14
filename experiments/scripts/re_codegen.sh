#!/bin/bash

# Usage: in `experiments`, run `bash re_codegen.sh $ROOT_DIR` where $ROOT_DIR is the parent folder of all good kernels (default: ./result)

ROOTDIR=${1:-./results}

echo "Using root directory: $ROOTDIR"

echo "Regenerating kernel scheduler directories for resnet good kernels......"
python analysis/re_codegen.py --model torchvision/resnet18 --batch-size 1 --kas-sampler-workers 400 \
 --kas-max-flops-ratio 0.9 --kas-min-flops-ratio 0.15 \
 --kas-max-enumerations 5 --kas-max-finalizations 2 --kas-depth 12 --kas-max-reductions 5 --kas-max-merges 1 --kas-max-splits 3 --kas-max-shifts 2 --kas-max-strides 0 --kas-max-size-multiplier 4 --kas-max-variables-in-size 3 --kas-max-chain-length 6 --kas-max-shift-rhs 2 --kas-max-expansion-merge-multiplier 2048 --kas-min-weight-share-dim 8 --kas-max-weight-share-dim 8 --kas-min-unfold-ratio 2.3 --client-mem-limit 1.0 --kas-scheduler-cache-dir .recodegen-scheduler-cache \
 --dirs $(find $ROOTDIR/resnet-good-kernels -mindepth 1 -maxdepth 1 -type d -regex ".*/0\.[0-9]x")

echo "Regenerating kernel scheduler directories for resnext good kernels......"
python analysis/re_codegen.py --model torchvision/resnext29_2x64d --batch-size 1 --kas-sampler-workers 400 \
 --kas-max-flops-ratio 0.95 --kas-min-flops-ratio 0.15 \
 --kas-max-enumerations 5 --kas-max-finalizations 2 --kas-depth 12 --kas-max-reductions 5 --kas-max-merges 2 --kas-max-splits 3 --kas-max-shifts 2 --kas-max-strides 0 --kas-max-size-multiplier 4 --kas-max-variables-in-size 3 --kas-max-chain-length 6 --kas-max-shift-rhs 2 --kas-max-expansion-merge-multiplier 4096 --kas-min-weight-share-dim 8 --kas-max-weight-share-dim 8 --kas-min-unfold-ratio 2.3 --client-mem-limit 1.0 --kas-scheduler-cache-dir .recodegen-scheduler-cache \
 --dirs $ROOTDIR/resnext-good-kernels

echo "Regenerating kernel scheduler directories for densenet good kernels......"
python analysis/re_codegen.py --model torchvision/densenet121 --batch-size 1 --kas-sampler-workers 400 \
 --kas-max-flops-ratio 1.0 --kas-min-flops-ratio 0.15 \
 --kas-max-enumerations 5 --kas-max-finalizations 2 --kas-depth 12 --kas-max-reductions 5 --kas-max-merges 1 --kas-max-splits 3 --kas-max-shifts 2 --kas-max-strides 0 --kas-max-size-multiplier 4 --kas-max-variables-in-size 3 --kas-max-chain-length 6 --kas-max-shift-rhs 2 --kas-max-expansion-merge-multiplier 5120 --kas-min-weight-share-dim 8 --kas-max-weight-share-dim 8 --kas-min-unfold-ratio 2.3 --client-mem-limit 1.0 --kas-scheduler-cache-dir .recodegen-scheduler-cache \
 --dirs $ROOTDIR/densenet-good-kernels

echo "Regenerating kernel scheduler directories for efficientnet good kernels......"
python analysis/re_codegen.py --model torchvision/efficientnet_v2_s --batch-size 1 --kas-sampler-workers 400 \
 --kas-max-flops-ratio 0.9 --kas-min-flops-ratio 0.5 \
 --kas-max-enumerations 5 --kas-max-finalizations 2 --kas-depth 12 --kas-max-reductions 5 --kas-max-merges 1 --kas-max-splits 3 --kas-max-shifts 2 --kas-max-strides 0 --kas-max-size-multiplier 4 --kas-max-variables-in-size 3 --kas-max-chain-length 6 --kas-max-shift-rhs 2 --kas-max-expansion-merge-multiplier 6144 --kas-min-weight-share-dim 8 --kas-max-weight-share-dim 24 --kas-min-unfold-ratio 2.3 --client-mem-limit 1.0 --kas-scheduler-cache-dir .recodegen-scheduler-cache \
 --dirs $ROOTDIR/efficientnet-good-kernels
