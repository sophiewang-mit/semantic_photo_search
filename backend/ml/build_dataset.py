import json, numpy as np, pandas as pd
from tqdm import tqdm
from app.embeddings import embed_text
from ml.features import make_features

IMG_VECS = np.load("data/image_embeddings.npy")
IMG_IDS = json.load(open("data/image_ids.json"))
# create look up table
ID_TO_ROW = {id: i for i, id in enumerate(IMG_IDS)}


# columns: image, caption
CAPTIONS_CSV = "data/flickr30k_subset/captions.csv"
N_HARD_NEG = 4
HELD_OUT = 100 #2000 #images reserved for the test set

def main():
    df = pd.read_csv(CAPTIONS_CSV)
    df = df[df["image"].isin(ID_TO_ROW)]

    # test and train on different samples
    test_imgs = set(IMG_IDS[-HELD_OUT:])
    train_df = df[~df["image"].isin(test_imgs)]

    X, y = [], []
    for _, row in tqdm(train_df.iterrows(), total = len(train_df), desc = "pairs"):
        img_id, caption = row["image"], row["caption"]
        if img_id not in ID_TO_ROW:
            continue
        t = embed_text(caption)
        t = t / np.linalg.norm(t)
        pos_row = ID_TO_ROW[img_id]

        # positive
        X.append(make_features(t, IMG_VECS[pos_row])); y.append(1)

        # hard negatives: calculate cosine similarity btwn query and every image
        sims = IMG_VECS @ t
        sims[pos_row] = -1.0
        # confusing wrong answers
        hard = np.argpartition(-sims, N_HARD_NEG)[:N_HARD_NEG]
        for neg_row in hard:
            X.append(make_features(t, IMG_VECS[neg_row])); y.append(0)
    
    X = np.stack(X).astype("float32")
    y = np.array(y, dtype = "float32")
    np.savez("data/train_pairs.npz", X=X, y=y)
    with open("data/test_images.json", "w") as f:
        json.dump(sorted(test_imgs), f)
    print(f"build {len(y)} pairs, {int(y.sum())} positive")

if __name__ == "__main__":
    main()