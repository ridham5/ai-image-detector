#!/usr/bin/env bash
# Run the whole pipeline on CIFAKE (same steps as the Colab notebook).
# Usage: bash scripts/run_all.sh            (assumes data/cifake exists)
set -euo pipefail
TRAIN=data/cifake/train
TEST=data/cifake/test
N_TRAIN=${N_TRAIN:-20000}   # images per class used for training
N_TEST=${N_TEST:-5000}      # images per class used for testing

python scripts/figures.py --data-dir $TEST
python scripts/train.py --method fft_svm    --train-dir $TRAIN --max-per-class 10000
python scripts/train.py --method resnet50   --train-dir $TRAIN --max-per-class $N_TRAIN --epochs 3
python scripts/train.py --method clip_probe --train-dir $TRAIN --max-per-class $N_TRAIN
for M in fft_svm resnet50 clip_probe; do
  python scripts/evaluate.py --method $M --test-dir $TEST --test-name cifake --max-per-class $N_TEST --robustness
done
# Generalisation (optional): put GenImage subsets in data/genimage/<generator>/val/{ai,nature}
if [ -d data/genimage ]; then
  for G in data/genimage/*/; do
    G=${G%/}; NAME=$(basename $G)
    for M in fft_svm resnet50 clip_probe; do
      python scripts/evaluate.py --method $M --test-dir $G/val --test-name $NAME --max-per-class 2000
    done
  done
fi
python scripts/make_tables.py --main-test cifake
python scripts/gradcam_figure.py --test-dir $TEST
