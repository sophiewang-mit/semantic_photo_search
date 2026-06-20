import json
import numpy as np
import pandas as pd, pickle
from tqdm import tqdm
from app.embeddings import embed_text
from ml.features import make_features
from ml.confidence import margin

IMG_VECS = np.load("data/image_embeddings.npy")
IMG_IDS = json.load(open("data/image_ids.json"))
ID_TO_ROW = {fid: i for i, fid in enumerate(IMG_IDS)}

clf=pickle.load(open("models/reranker.pkl", "rb"))

# number of candidates CLIP passes to the reranker
CAPTIONS_CSV = "data/flickr30k_subset/captions.csv"
TOP_N = 100
KS = [1, 5, 10]

def dcg_at_k(rank, k):
    return 1.0 / np.log2(rank + 1) if rank <= k else 0.0

def metrics_init():
    m = {f"R@{k}": 0 for k in KS}
    m["MRR"] = 0.0
    m["nDCG@10"] = 10
    return m

def accumulate(m, rank):
    for k in KS:
        m[f"R@{k}"] += int(rank <= k)
    if rank < 10**9:
        m["MRR"] += 1.0 / rank
    m["nDCG@10"] += dcg_at_k(rank, 10)

def run(df, image_set, threshold):
    """Evaluate on image set.
    Rerank only when CLIP is uncertain:
        margin(cand_scores) < threshold
    threshold = -np.inf -> dont rank, just CLIP
    threshold = np.inf -> rerank"""
    rows = df[df["image"].isin(image_set)]

    pool_rows = [ID_TO_ROW[img_id] for img_id in IMG_IDS if img_id in image_set]
    sub_vecs = IMG_VECS[pool_rows]
    row_to_id = [IMG_IDS[r] for r in pool_rows]

    m = metrics_init()
    n = 0
    reranked = 0


    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"thr={threshold}"):
        true_id = row["image"]

        t = embed_text(row["caption"])
        t = t / np.linalg.norm(t)

        sims = sub_vecs @ t

        topn = min(TOP_N, len(sims))
        cand = np.argpartition(-sims, topn - 1)[:topn]
        cand = cand[np.argsort(-sims[cand])]

        cand_scores = sims[cand]

        if margin(cand_scores) < threshold:
            feats = np.stack([make_features(t, sub_vecs[c]) for c in cand])
            probs = clf.predict_proba(feats)[:, 1]
            order = cand[np.argsort(-probs)]
            reranked += 1
        else:
            order = cand

        ordered_ids = [row_to_id[c] for c in order]

        if true_id in ordered_ids:
            rank = ordered_ids.index(true_id) + 1
        else:
            rank = 10**9

        accumulate(m, rank)
        n += 1

    for key in m:
        m[key] /= n

    return m, reranked / n


def print_table(results):
    df = pd.DataFrame(results).T
    print(df.round(4))


def main():
    df = pd.read_csv(CAPTIONS_CSV)

    val_imgs = set(json.load(open("data/val_images.json")))
    test_imgs = set(json.load(open("data/test_images.json")))

    thresholds = [
        -np.inf,
        0.01,
        0.02,
        0.05,
        0.08,
        0.10,
        0.15,
        np.inf,
    ]

    print("\nValidation results:")
    val_results = {}

    best_threshold = None
    best_score = -1

    for threshold in thresholds:
        m, frac = run(df, val_imgs, threshold)
        m["frac_reranked"] = frac
        val_results[str(threshold)] = m

        if m["R@1"] > best_score:
            best_score = m["R@1"]
            best_threshold = threshold

    print_table(val_results)

    print(f"\nBest threshold on validation: {best_threshold}")

    print("\nFinal test results:")
    test_results = {}

    for name, threshold in {
        "CLIP only": -np.inf,
        "Always rerank": np.inf,
        "Gated rerank": best_threshold,
    }.items():
        m, frac = run(df, test_imgs, threshold)
        m["frac_reranked"] = frac
        test_results[name] = m

    print_table(test_results)


if __name__ == "__main__":
    main()

