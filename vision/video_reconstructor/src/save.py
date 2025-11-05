import cv2

# ==========================================================
#  Save Cleaned Video
# ==========================================================
def save_video(frames, out_path, fps):
    """
    Writes and saves the reconstructed video.

    """
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"Video successfully saved to {out_path}")
    