import json, numpy as np, pandas as pd
from ml.evaluate_gated import run

def main():
    df = pd.read_csv("data/flickr30k_subset/captions.csv")
    val = set(json.load(open("data/val_images.json")))
    test = set(json.load(open("data/test_images.json")))

    # candidate thresholds to try on VALIDATION only
    grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20]

    best_thr, best_ndcg = None, -1.0
    print("validation sweep:")
    for thr in grid:
        m, frac = run(df, val, thr)
        print(f"  thr={thr:.3f}  nDCG@10={m['nDCG@10']:.4f}  reranked={frac:.1%}")
        if m["nDCG@10"] > best_ndcg:
            best_ndcg, best_thr = m["nDCG@10"], thr
    print(f"\npicked threshold = {best_thr} (val nDCG@10={best_ndcg:.4f})")

    # frozen comparisons on TEST
    clip_only, _ = run(df, test, float("-inf"))
    always, f_always = run(df, test, float("inf"))
    gated, f_gated = run(df, test, best_thr)

    print("\n=== TEST SET (frozen threshold) ===")
    print(f"{'metric':10s}{'CLIP':>10s}{'always-rr':>12s}{'gated':>10s}")
    for key in clip_only:
        print(f"{key:10s}{clip_only[key]:>10.4f}{always[key]:>12.4f}{gated[key]:>10.4f}")
    print(f"\nreranked fraction: always={f_always:.1%} gated={f_gated:.1%}")

if __name__ == "__main__":
    main()