from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil, os
from .embeddings import embed_image, embed_text
from .search import PhotoIndex
from .reranker import Reranker
from fastapi.staticfiles import StaticFiles

# server
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# upload folders
UPLOAD_DIR = "uploads"
DEMO_DIR = "demo_images"

os.makedirs(UPLOAD_DIR, exist_ok= True)
os.makedirs(DEMO_DIR, exist_ok= True)

#serve image files
app.mount("/uploads", StaticFiles(directory = UPLOAD_DIR), name = "uploads")
app.mount("/demo_images", StaticFiles(directory=DEMO_DIR), name = "demo_images")

# image database
index = PhotoIndex()
reranker = Reranker()

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webg", ".avif")

# index images in demo_images
def index_folder(folder):
    for filename in os.listdir(folder):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue
        path = os.path.join(folder, filename)
        try:
            index.add(embed_image(path), path)
        except Exception as e:
            print(f"Failed to index {path}: {e}")

@app.on_event("startup")
def load_demo_images():
    index_folder(DEMO_DIR)

@app.post("/upload")
# uploads image
async def upload(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    index.add(embed_image(path), path)
    return {"status": "indexed", "filename": file.filename}

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