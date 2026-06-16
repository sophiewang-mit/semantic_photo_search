import numpy as np, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def main():
    d = np.load("data/train_pairs.npz")
    # features and labels matrices
    X = d["X"]
    y = d["y"]

    #randomize 
    idx = np.random.RandomState(0).permutation(len(y))
    cut = int(0.9 * len(y))
    tr, va = idx[:cut], idx[cut:]

    # logistic training regression
    clf = LogisticRegression(max_iter = 1000, class_weight = "balanced")
    clf.fit(X[tr], y[tr])

    auc = roc_auc_score(y[va], clf.predict_prob(X[va]))[:, 1]
    print(f"validation pair-AUC: {auc:.4f}")

    with open("models/reranker.pkl", "wb") as f:
        pickle. dump(clf, f)
    print("saved models/reranked.pkl")

if __name__ == "__main__":
    main()