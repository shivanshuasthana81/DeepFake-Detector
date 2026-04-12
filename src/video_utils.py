# src/video_utils.py
import cv2
import os
import numpy as np

def extract_frames(video_path, out_dir, fps=2):
    """
    Extract frames at approx `fps` frames per second.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video: " + video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(orig_fps / fps))
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            fname = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
        idx += 1
    cap.release()
    return saved
