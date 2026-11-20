#!/usr/bin/env bash
# Full run on a Mac (CPU for ResNet-50). Usage: bash scripts/run_overnight.sh
# Keeps the Mac awake (caffeinate) and stops at the first error.
set -e
cd "$(dirname "$0")/.."
caffeinate -i -w $$ &
echo "Started: $(date)"
python scripts/train.py --method fft_svm --train-dir data/cifake/train --max-per-class 10000
python scripts/train.py --method resnet50 --train-dir data/cifake/train --max-per-class 10000 --epochs 2 --batch-size 32 --img-size 160 --num-workers 0 --device cpu
python scripts/train.py --method clip_probe --train-dir data/cifake/train --max-per-class 10000 --batch-size 32 --num-workers 0
python scripts/evaluate.py --method fft_svm --test-dir data/cifake/test --test-name cifake --max-per-class 2000 --robustness
python scripts/evaluate.py --method resnet50 --test-dir data/cifake/test --test-name cifake --max-per-class 2000 --robustness --num-workers 0 --device cpu
python scripts/evaluate.py --method clip_probe --test-dir data/cifake/test --test-name cifake --max-per-class 2000 --robustness --num-workers 0
python scripts/make_tables.py --main-test cifake
python scripts/gradcam_figure.py --test-dir data/cifake/test --n 3 --device cpu
echo "ALL DONE: $(date)"
