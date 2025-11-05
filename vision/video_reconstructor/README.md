# Video Reconstruction with CLIP, HDBSCAN and TSP Optimization

## Description

This project implements a fully unsupervised pipeline for **video cleaning and temporal reordering** using **CLIP embeddings**, **HDBSCAN clustering**, and **graph-based optimization**.  
Given a potentially corrupted or shuffled video, the system extracts semantically meaningful frame embeddings, filters out noise, and reconstructs a visually coherent sequence by solving a **Traveling Salesman Problem (TSP)** on the similarity graph.

The approach leverages **OpenAI’s CLIP** model to capture high-level visual semantics, and applies **HDBSCAN** to isolate the main visual context before ordering frames through nearest-neighbor heuristics and **2-Opt refinement**.




## Features
    
-   **CLIP-Based Feature Extraction**: Leverages OpenAI CLIP for semantically meaningful frame representations.
    
-   **Unsupervised Frame Clustering**: HDBSCAN identifies dominant visual contexts automatically.
    
-   **Graph-Based Temporal Ordering**: Combines nearest-neighbor heuristics and 2-Opt TSP refinement for optimal sequencing.
    
-   **Automatic Frame Sampling and Reconstruction**: Efficient preprocessing and video writing pipeline.

-   **Efficiency**: Frame sampling and batched embedding extraction ensure scalability to long videos.
    
-   **Fully Modular Implementation**: Clean, extensible NumPy/PyTorch codebase.


## Methodology

### 1. Frame Sampling

Frames are sampled from the input video at fixed stride intervals to obtain at most _N_ representative frames.

```python
frames = sample_frames(video_path, max_frames=500)
```

### 2. Embedding Extraction

Each frame is encoded using a pretrained **CLIP model** (`ViT-B/32` by default), producing normalized semantic embeddings.

```python
model, preprocess, device = load_clip_model("ViT-B/32")
embs = extract_embeddings(frames, model, preprocess, device, batch_size=64)
```

### 3. Clustering

**HDBSCAN** identifies the dominant visual context by discarding noisy clusters.

```python
labels = cluster_embeddings(embs, min_cluster_size=5)
main_idx = filter_embeddings(labels)
```

### 4. Temporal Ordering

Frames are re-ordered by minimizing cosine distance across embeddings. A **nearest-neighbor heuristic** combined with **2-Opt refinement** approximates the optimal frame sequence.

```python
ordered_indices = order_frames(embs, main_idx, n_restarts=20)
```

### 5. Reconstruction

The ordered frames are written back to a cleaned video file.

```python
save_video(ordered_frames, "reconstructed_video.mp4", fps=25)
```


## File Structure

```
├── main.py — Full cleaning pipeline
│ 
├── requirements.txt — Lists all the Python packages needed
│
└── src/ — Core source code  
  ├── sample.py — Implements frame sampling with a fixed stride 
  ├── clip.py — Imports CLIP and extracts embeddings
  ├── hdbscan.py — Clusters and filter the frames
  ├── order.py — Orders frames using NN + 2-Opt
  └── save.py — Saves cleaned video
```
        


## Installation


`pip install -r requirements.txt` 


## Usage

Run the reconstruction pipeline:

```bash
python main.py
```

Default parameters:

```python
video_path = "corrupted_video.mp4"
out_path   = "reconstructed_video.mp4"
```

The cleaned and reordered video will be saved as `reconstructed_video.mp4`.



## References

-   **CLIP**: _Learning Transferable Visual Models From Natural Language Supervision_ — Radford et al., 2021
    
-   **HDBSCAN**: _Robust Clustering by Density Estimation_ — Campello et al., 2013
    
-   **2-Opt Algorithm**: Classic TSP local search optimization