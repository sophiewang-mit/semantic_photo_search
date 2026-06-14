import os, json
import numpy as np
from tqdm import tqdm
from app.embeddings import embed_image

IMAGE_DIR = "data/flickr30k_subset/images"
OUT_VECS = "data/image_embeddings.npy"
OUT_IDS = "data/image_ids.json"

def main():
    files = sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    vectors, ids = [], []
    # loop over every image
    for fname in tqdm(files, desc = "embedding images"):
        try:
            vec = embed_image(os.path.join(IMAGE_DIR, fname)) # CLIP
            vectors.append(vec)
            ids.append(fname)
        except Exception as e:
            print(f"skip {fname}: {e}")
    arr = np.stack(vectors).astype("float32")
    # normalize 
    arr /= np.linalg.norm(arr, axis = 1, keepdims = True)
    np.save(OUT_VECS, arr)
    with open(OUT_IDS, "w") as f:
        json.dump(ids, f)
    print(f"saved {arr.shape[0]} embeddings, dim {arr.shape[1]}")

if __name__ == "__main__":
    main()