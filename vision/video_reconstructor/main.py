import cv2

from src.clip import load_clip_model, extract_embeddings
from src.hdbscan import cluster_embeddings, filter_embeddings
from src.order import order_frames
from src.sample import sample_frames
from src.save import save_video

# ==========================================================
#  Main Cleaning Pipeline
# ==========================================================
def clean_video(video_path,
             out_path,
             max_sampled_frames=500,
             clip_model_type="ViT-B/32",
             batch_size_clip=64,
             min_cluster_size=5,
             n_restarts=20,
             fps_out=25
             ):
    
    print("1) Sampling frames ...")
    frames_meta = sample_frames(video_path, max_sampled_frames)
    frames = [cv2.cvtColor(f[0], cv2.COLOR_BGR2RGB) for f in frames_meta]
    
    print("\n2) Loading CLIP model...")
    model, preprocess, device = load_clip_model(clip_model_type)
    
    print("\n3) Extracting CLIP embeddings...")
    embs = extract_embeddings(frames, model, preprocess, device, batch_size_clip)
        
    print("\n4) Clustering embeddings with HDBSCAN...")
    labels = cluster_embeddings(embs, min_cluster_size)
    
    print("\n5) Filtering embeddings by keeping the largest cluster...")
    main_idx = filter_embeddings(labels)
    
    print("\n6) Ordering frames with cosine similarity graph ...")
    ordered_indices = order_frames(embs, main_idx, n_restarts)

    print("\n7) Saving the reconstructed video...")
    ordered_frames = [frames_meta[idx][0] for idx in ordered_indices]
    save_video(ordered_frames, out_path, fps=fps_out)

        
if __name__ == "__main__":
    video_path = "corrupted_video.mp4"
    out_path = "reconstructed_video.mp4"
    clean_video(video_path, out_path)