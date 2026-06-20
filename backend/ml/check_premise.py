import json, numpy as np, pandas as pd
from tqdm import tqdm
from app.embeddings import embed_text
from ml.confidence import margin

IMG_VECS = np.load("data/image_embeddings.npy")
IMG_IDS = json.load(open("data/image_ids.json"))
ID_TO_ROW = {fid: i for i, fid in enumerate(IMG_IDS)}
val = set(json.load(open("data/val_images.json")))

def main():
    df = pd.read_csv("data/flickr30k_subset/captions.csv")
    rows = df[df["image"].isin(val)]
    pool_rows = [ID_TO_ROW[i] for i in IMG_IDS if i in val]
    sub_vecs = IMG_VECS[pool_rows]; row_to_id = [IMG_IDS[r] for r in pool_rows]

    margins, correct = [], []
    for _, row in tqdm(rows.iterrows(), total=len(rows)):
        t = embed_text(row["caption"]); t /= np.linalg.norm(t)
        sims = sub_vecs @ t
        cand = np.argsort(-sims)
        margins.append(margin(sims[cand]))
        correct.append(int(row_to_id[cand[0]] == row["image"]))  # CLIP top-1 right?

    margins, correct = np.array(margins), np.array(correct)
    lo = margins < np.median(margins)   # low-confidence half
    print(f"CLIP top-1 accuracy, LOW-margin half:  {correct[lo].mean():.3f}")
    print(f"CLIP top-1 accuracy, HIGH-margin half: {correct[~lo].mean():.3f}")

if __name__ == "__main__":
    main()