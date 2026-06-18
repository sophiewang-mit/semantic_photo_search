import json, numpy as np, pandas as pd, pickle
from tqdm import tqdm
from app.embeddings import embed_text
from ml.features import make_features

IMG_VECS = np.load("data/image_embeddings.npy")
IMG_IDS = json.load(open("data/image_ids.json"))
ID_TO_ROW = {fid: i for i, fid in enumerate(IMG_IDS)}
# load test image split and trained logistic regression reranker
test_imgs = set(json.load(open("data/test_images.json")))
clf=pickle.load(open("models/reranker.pkl", "rb"))

# number of candidates CLIP passes to the reranker
TOP_N = 100
KS = [1, 5, 10]

def dcg_at_k(rank, k):
    return 1.0 / np.log2(rank + 1) if rank <= k else 0.0

def evaluate(df):
    # only keep captions of images' who are in the test set
    rows = df[df["image"].isin(test_imgs)]
    # check if correct image was within first k results
    base = {f"R@{k}": 0 for k in KS}; base["MRR"] = 0.0; base["nDCG@10"] = 0.0
    rer = {f"R@{k}": 0 for k in KS}; rer["MRR"] = 0.0; rer["nDCG@10"] = 0.0
    n=0

    # test_rows = [ID_TO_ROW[i] for i in IMG_IDS if i in test_imgs]
    sub_vecs = IMG_VECS #IMG_VECS[test_rows]
    row_to_id = IMG_IDS #[IMG_IDS[r] for r in test_rows]

    for _, row in tqdm(rows.iterrows(), total = len(rows), desc = "eval"):
        true_id = row["image"]
        # embed query (caption) + compare with imgaes + take 100 top most similar
        query_text = row["caption"]
        t = embed_text(query_text)
        t /= np.linalg.norm(t)

        sims = sub_vecs @ t
        k = min(TOP_N, len(sims))
        cand = np.argsort(-sims)[:k]

        if n == 0:
            print("Candidate pool size:", len(sims))
            print("reranker candidate size:", len(cand))
            
        cand=cand[np.argsort(-sims[cand])]

        # iniitally CLIP rank
        clip_order = [row_to_id[c] for c in cand]
        # reranked order: probability image matches caption
        feats = np.stack([make_features(t, sub_vecs[c]) for c in cand])
        scores = clf.predict_proba(feats)[:, 1]
        # order based on reranker
        rer_order = [row_to_id[c] for c in cand[np.argsort(-scores)]]

        # find rank of image
        def rank_of(order):
            return order.index(true_id) + 1 if true_id in order else 10**9
        
        for name, order, acc in [("clip", clip_order, base), ("rer", rer_order, rer)]:
            r = rank_of(order)
            for k in KS:
                acc[f"R@{k}"] += int(r <=k)
            acc["MRR"] += 1.0 / r if r < 10**9 else 0.0
            acc["nDCG@10"] += dcg_at_k(r, 10)
        n+=1

    for acc in (base, rer):
        for key in acc:
            acc[key] /= n
    return base, rer

# higher = greater similarity
if __name__ == "__main__":
    df = pd.read_csv("data/flickr30k_subset/captions.csv")
    base, rer= evaluate(df)
    print("\n       CLIP        CLIP + reranker")
    for key in base:
        print(f"{key:8s}    {base[key]:.4f}     {rer[key]:.4f}")