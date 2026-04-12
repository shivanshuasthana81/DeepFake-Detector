# src/utils/face_detect.py
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import cv2
import os

mtcnn = MTCNN(keep_all=True)

def crop_faces_from_frame(frame, size=(224,224)):
    """
    frame: BGR numpy image (as from cv2)
    returns list of face images (PIL) resized to size
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    boxes, _ = mtcnn.detect(pil)
    crops = []
    if boxes is None:
        return crops
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        face = pil.crop((x1, y1, x2, y2)).resize(size)
        crops.append(face)
    return crops

def save_crops_from_video(video_path, out_dir, frames_to_sample=10, size=(224,224)):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // frames_to_sample)
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            faces = crop_faces_from_frame(frame, size=size)
            for i, face in enumerate(faces):
                fname = os.path.join(out_dir, f"{os.path.basename(video_path)}_f{idx}_{i}.jpg")
                face.save(fname)
                saved += 1
        idx += 1
    cap.release()
    return saved
