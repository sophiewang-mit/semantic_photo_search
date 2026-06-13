from sentence_transformers import SentenceTransformer
from PIL import Image 

# CLIP model that includes images and text into the same space
model = SentenceTransformer("clip-ViT-B-32")

def embed_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    return model.encode(img, convert_to_numpy=True)

def embed_text(text: str):
    return model.encode(text, convert_to_numpy=True)