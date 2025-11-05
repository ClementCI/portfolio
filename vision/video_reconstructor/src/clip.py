import torch
import clip
import numpy as np

from PIL import Image
from tqdm import tqdm

# ==========================================================
#  CLIP Embeddings
# ==========================================================
def load_clip_model(model_type):
    """
    Loads CLIP model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_type, device=device)
    model.eval()
    return model, preprocess, device

def extract_embeddings(frames, model, preprocess, device, batch_size):
    """
    Extracts frames' embeddings using CLIP model.
    """
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(frames), batch_size), desc="Extracting CLIP embeddings"):
            frames_batch = frames[i:i+batch_size]
            pil_frames_batch = [Image.fromarray(fr) for fr in frames_batch]              # convert frames to PIL images
            inputs = torch.stack([preprocess(fr) for fr in pil_frames_batch]).to(device) # preprocess PIL frames and stack
            embs_batch = model.encode_image(inputs)                                      # encode
            embs_batch = embs_batch / embs_batch.norm(dim=-1, keepdim=True)              # normalize
            embs.append(embs_batch.cpu().numpy())                                        # back to cpu and numpy
    return np.vstack(embs)
