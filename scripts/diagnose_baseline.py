"""
Phase 0 diagnostic visualisation of the official Upright-Net baseline.

Reads logs/baseline_official.npz (produced by eval_official_baseline.py) and
produces:
  - logs/baseline_error_cdf.png      : CDF + histogram + τ sweep (Pang Fig.4-like)
  - logs/baseline_per_category.csv   : per-class mean_err, acc@10, flip rate
  - logs/baseline_failure_samples.txt: top-30 worst samples with metadata
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "logs" / "baseline_official.npz"

CATEGORY_NAMES = [
    "bed", "bench", "bottle", "bowl", "car",
    "chair", "cone", "cup", "lamp", "monitor",
    "sofa", "stool", "table", "toilet", "vase",
]  # From datasets/uprightnet15/shapename.txt


def main():
    d = np.load(LOG)
    err = d["error_fix_deg"]  # honest version (no GT leak)
    err_leak = d["error_leak_deg"]
    label = d["label"].squeeze()
    nsup = d["num_support"]
    print(f"Loaded {len(err)} samples, {len(np.unique(label))} unique labels")

    # --- Plot 1: CDF + τ sweep + histogram ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # τ sweep (Pang Fig.4-style accuracy curve)
    tau = np.arange(0, 181, 1)
    acc_curve = np.array([(err < t).mean() for t in tau]) * 100
    axes[0].plot(tau, acc_curve, linewidth=2, color="tab:blue",
                 label=f"Upright-Net (official)\nmean={err.mean():.2f}°")
    axes[0].axvline(10, ls="--", color="gray", alpha=0.5, label="τ=10°")
    axes[0].axvline(90, ls="--", color="tab:red", alpha=0.5, label="τ=90° (flip)")
    axes[0].axvline(180, ls="--", color="gray", alpha=0.3)
    axes[0].set_xlabel("τ (degrees)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy vs angular threshold (Pang Fig.4)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # Histogram of errors (log scale)
    axes[1].hist(err, bins=np.linspace(0, 180, 91), log=True, color="tab:blue")
    axes[1].axvline(90, ls="--", color="tab:red", alpha=0.5)
    axes[1].set_xlabel("Angular error (degrees)")
    axes[1].set_ylabel("# samples (log)")
    axes[1].set_title("Error distribution")
    axes[1].grid(alpha=0.3)

    # Zoomed: 90-180 tail
    axes[2].hist(err[err > 30], bins=np.linspace(30, 180, 76), color="tab:red")
    axes[2].axvline(90, ls="--", color="black", alpha=0.5, label="flip threshold")
    axes[2].set_xlabel("Angular error (degrees)")
    axes[2].set_ylabel("# samples")
    axes[2].set_title(f"Tail of error distribution (err > 30°)\n"
                      f"{(err > 90).sum()} flips / {len(err)} total = "
                      f"{(err > 90).mean()*100:.2f}%")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out1 = REPO / "logs" / "baseline_error_cdf.png"
    plt.savefig(out1, dpi=120)
    print(f"Saved {out1}")

    # --- Per-category breakdown ---
    print()
    print(f"{'label':>6}  {'N':>6}  {'mean':>7}  {'median':>7}  "
          f"{'acc@10':>7}  {'flip%':>7}  {'near180%':>9}")
    rows = []
    for lab in sorted(np.unique(label)):
        m = label == lab
        e = err[m]
        mean = e.mean()
        med = np.median(e)
        acc10 = (e < 10).mean() * 100
        flip = (e > 90).mean() * 100
        near180 = ((e > 170)).mean() * 100
        cat = CATEGORY_NAMES[int(lab)] if int(lab) < len(CATEGORY_NAMES) else "?"
        print(f"{int(lab):>3} {cat:>10}  N={m.sum():>5}  mean={mean:6.2f}  "
              f"med={med:6.2f}  acc@10={acc10:6.2f}  "
              f"flip>90°={flip:6.2f}  near180°={near180:6.2f}")
        rows.append((int(lab), cat, m.sum(), mean, med, acc10, flip, near180))

    # Save CSV
    out_csv = REPO / "logs" / "baseline_per_category.csv"
    with open(out_csv, "w") as f:
        f.write("label,category,N,mean_err,median_err,acc10,flip_rate,near180_rate\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"Saved {out_csv}")

    # --- Top-30 worst samples ---
    idx_sorted = np.argsort(-err)
    out_txt = REPO / "logs" / "baseline_failure_samples.txt"
    with open(out_txt, "w") as f:
        f.write("# Top-30 worst predictions by official Upright-Net\n")
        f.write("# idx, label, err_deg, num_support\n")
        for k in idx_sorted[:30]:
            cat = CATEGORY_NAMES[int(label[k])] if int(label[k]) < len(CATEGORY_NAMES) else "?"
            f.write(f"{int(k):7d}  lab={int(label[k]):2d} ({cat:>10})  "
                    f"err={err[k]:7.2f}°  n_sup={int(nsup[k]):4d}\n")
    print(f"Saved {out_txt}")

    # --- Summary of flip-vs-nonflip ---
    print()
    print("=" * 72)
    print(f"Total samples: {len(err)}")
    print(f"  err < 5°:        {(err < 5).sum():6d}  ({(err < 5).mean()*100:5.2f}%)")
    print(f"  err in [5,10)°:  {((err>=5)&(err<10)).sum():6d}  ({((err>=5)&(err<10)).mean()*100:5.2f}%)")
    print(f"  err in [10,30)°: {((err>=10)&(err<30)).sum():6d}  ({((err>=10)&(err<30)).mean()*100:5.2f}%)")
    print(f"  err in [30,90)°: {((err>=30)&(err<90)).sum():6d}  ({((err>=30)&(err<90)).mean()*100:5.2f}%)")
    print(f"  err > 90° (flip):{(err>=90).sum():6d}  ({(err>=90).mean()*100:5.2f}%)")
    print(f"  err > 170° (near-perfect flip):{(err>170).sum():6d}  ({(err>170).mean()*100:5.2f}%)")
    print()
    print("Interpretation:")
    print(f"  - Non-flip error median = {np.median(err[err < 90]):.2f}°")
    print(f"  - Flip error median     = {np.median(err[err >= 90]):.2f}°")
    print(f"  - Flip rate is {(err > 90).mean()*100:.2f}% -> this is the 'τ=180° boost' of Pang Fig.4")


if __name__ == "__main__":
    main()
