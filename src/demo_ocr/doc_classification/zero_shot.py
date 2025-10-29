from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Iterable

from .ml_caller.openclip_backend import get_openclip_backend


def crop_top_percent(img: np.ndarray, top_percent: float) -> np.ndarray:
    """
    Crop the top portion of an image by a given percentage of its height (full width).

    Parameters
    ----------
    img : np.ndarray
        Image array with shape (H, W) or (H, W, C). dtype ใดก็ได้ (uint8, float32, ฯลฯ)
    top_percent : float
        ถ้า 0 < top_percent <= 1.0 จะถูกตีความว่าเป็นสัดส่วน (เช่น 0.35 = 35%)
        ถ้า top_percent > 1.0 จะถูกตีความว่าเป็นเปอร์เซ็นต์ (เช่น 35 = 35%)
        ค่าจะถูก clamp ให้อยู่ในช่วง [0, 1]

    Returns
    -------
    np.ndarray
        ภาพที่ถูกครอปด้านบนเต็มความกว้าง สูงเท่ากับ n% ของความสูงเดิม
        (คืนค่าเป็น view; ถ้าต้องการสำเนาให้ต่อด้วย .copy())
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy ndarray")
    if img.ndim not in (2, 3):
        raise ValueError("img must have shape (H, W) or (H, W, C)")

    H = img.shape[0]
    # แปลงเป็นสัดส่วน 0..1
    ratio = top_percent if 0.0 <= top_percent <= 1.0 else (top_percent / 100.0)
    ratio = float(np.clip(ratio, 0.0, 1.0))

    crop_h = int(round(H * ratio))
    # คืนอาเรย์ว่างถ้า 0% (ไม่ error)
    return img[:crop_h, ...]


def load_centroids_dict(path: Path | str) -> Dict[str, np.ndarray]:
    blob = np.load(str(path), allow_pickle=True)
    labels = blob["labels"].tolist()
    C = blob["centroids"]
    return {lbl: C[i].astype(np.float32) for i, lbl in enumerate(labels)}


def classify_open_set(
    q_vecs: Iterable[np.ndarray],
    centroids: Dict[str, np.ndarray],
    tau_per_label: Dict[str, float],
    margin: float,
    default_tau: float = 0.85,
) -> dict:
    q_vecs = list(q_vecs)
    if not q_vecs:
        raise ValueError("q_vecs is empty")
    if not centroids:
        raise ValueError("no centroids provided")

    scores = {
        lab: max(float(q @ c) for q in q_vecs)
        for lab, c in centroids.items()
    }
    top = sorted(scores.items(), key=lambda x: -x[1])
    lab1, s1 = top[0]
    tau = tau_per_label.get(lab1, default_tau)

    if len(top) == 1:
        # ไม่มีคู่แข่ง → margin ไม่เกี่ยว
        pred = lab1 if s1 >= tau else "UNKNOWN"
        return {"pred": pred, "s1": s1, "s2": -1.0, "scores": scores, "top": top}

    s2 = top[1][1]
    if s1 < tau or (s1 - s2) < margin:
        return {"pred": "UNKNOWN", "s1": s1, "s2": s2, "scores": scores, "top": top}
    return {"pred": lab1, "s1": s1, "s2": s2, "scores": scores, "top": top}


class ZeroShotCentroidClassifier:
    def __init__(self, input_centroid: dict,
                 input_tau: dict,
                 input_clip_model_name: str,
                 input_clip_pretrained: str
                 ):
        self.backend = get_openclip_backend(
            input_clip_model_name=input_clip_model_name, input_clip_pretrained=input_clip_pretrained)
        self.centroids = input_centroid
        self.tau = input_tau
        self.margin = 0.03

    def predict_from_rgb(self, rgb_uint8: np.ndarray,
                         scales: Iterable[float] = (0.8, 1.0, 1.25)) -> dict:
        q_vecs = self.backend.embed_multiscale(rgb_uint8, scales=scales)
        return classify_open_set(q_vecs, self.centroids, self.tau, margin=self.margin)


_classifier_singleton: ZeroShotCentroidClassifier | None = None


def get_classifier(input_centroid: dict,
                   input_tau: dict,
                   input_clip_model_name: str,
                   input_clip_pretrained: str) -> ZeroShotCentroidClassifier:
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = ZeroShotCentroidClassifier(
            input_centroid=input_centroid,
            input_tau=input_tau,
            input_clip_model_name=input_clip_model_name,
            input_clip_pretrained=input_clip_pretrained)
    return _classifier_singleton
