import cv2

from tqdm import tqdm

# ==========================================================
#  Frame sampling
# ==========================================================
def sample_frames(video_path, max_frames):
    """
    Samples frames from video with a fixed stride to get, at most, 'max_frames' frames.

    """
    # Open the video and get total number of frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Compute stride to get approximately at most max_frames
    if total_frames <= max_frames:
        stride = 1
    else:
        stride = max(1, round(total_frames / max_frames))
    print(f"Sampling every {stride} frames (~{total_frames // stride} frames total)")
    
    # Read the video and sample frames
    frames = []
    idx = 0
    saved = 0
    pbar = tqdm(total=total_frames, desc="Reading frames", unit="fr")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append((frame.copy(), idx))
            saved += 1
        idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    return frames