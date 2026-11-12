"""Evaluate a trained detector on a test folder, optionally under several perturbations.
Each (perturbation) run appends one row to results/results.csv.

Examples
    # main result on CIFAKE test set
    python scripts/evaluate.py --method resnet50 --test-dir data/cifake/test --test-name cifake

    # robustness: same test set, many perturbations
    python scripts/evaluate.py --method resnet50 --test-dir data/cifake/test --test-name cifake --robustness

    # generalisation: a different generator from GenImage
    python scripts/evaluate.py --method resnet50 --test-dir data/genimage/midjourney/val --test-name midjourney
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import list_images  # noqa: E402
from src.detectors import DEFAULT_FILES, METHODS, load_detector  # noqa: E402
from src.metrics import append_result, compute_metrics, pretty  # noqa: E402
from src.perturb import DEFAULT_SUITE, get_perturbation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--model", default=None, help="default: models/<method>.*")
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--test-name", required=True, help="short name for the table, e.g. cifake, midjourney")
    ap.add_argument("--train-name", default="cifake")
    ap.add_argument("--perturb", nargs="*", default=["none"], help="e.g. none jpeg:75 blur:1")
    ap.add_argument("--robustness", action="store_true", help=f"run the full suite: {DEFAULT_SUITE}")
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--save-preds", action="store_true", help="also save per-image probabilities")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="force a device (on a Mac, cpu avoids MPS training problems)")
    a = ap.parse_args()
    if a.device != "auto":
        import os
        os.environ["DETECTOR_DEVICE"] = a.device

    model_path = a.model or f"models/{DEFAULT_FILES[a.method]}"
    det = load_detector(a.method, model_path)
    items = list_images(a.test_dir, a.max_per_class, a.seed)
    paths, y = [p for p, _ in items], np.array([l for _, l in items])
    print(f"{a.method} on {a.test_name}: {len(items)} images ({y.sum()} fake)")

    specs = DEFAULT_SUITE if a.robustness else a.perturb
    for spec in specs:
        t = time.time()
        pert = get_perturbation(spec)
        if a.method == "fft_svm":
            p = det.predict_proba_paths(paths, pert)
        else:
            p = det.predict_proba_paths(paths, pert, batch_size=a.batch_size, num_workers=a.num_workers)
        m = compute_metrics(y, p)
        print(f"  [{spec:>10}] {pretty(m)}  ({time.time() - t:.0f}s)")
        append_result(a.results, {"method": a.method, "train_set": a.train_name, "test_set": a.test_name,
                                  "perturbation": spec, **m})
        if a.save_preds:
            out = Path(a.results).parent / "preds" / f"{a.method}__{a.test_name}__{spec.replace(':', '')}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                f.write("path,label,p_fake\n")
                for q, l, pp in zip(paths, y, p):
                    f.write(f"{q},{l},{pp:.6f}\n")


if __name__ == "__main__":
    main()
