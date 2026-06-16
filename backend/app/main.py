from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil, os
from .embeddings import embed_image, embed_text
from .search import PhotoIndex
from .reranker import Reranker

# server
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins = ['https://localhost:5173'], allow_methods = ["*"], allow_headers = ["*"],)

# image database
index = PhotoIndex()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok = True)

@app.post("/upload")
# uploads image
async def upload(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    index.add(embed_image(path), path)
    return {"status": "indexed", "filename": file.filename}

reranker = Reranker()
# search through query
@app.get("/search")
async def search(q: str, k: int = 12, rerank: bool = True):
    qvec = embed_text(q)

    candidates = index.search_with_vectors(qvec, k = 100)

    if rerank:
        pairs = [(p, v) for p, _, v in candidates]
        ranked = reranker.rerank(qvec, pairs)[: k]

        return {"results": [{"path": p, "score": s} for p, s in ranked], "mode": "reranked"}

    return {"results": [{"path": p, "score": s} for p, s, _ in candidates[:k]], "mode": "clip"}