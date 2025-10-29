# app/modules/openclip_backend.py
from __future__ import annotations
import numpy as np
import torch
import open_clip
from PIL import Image
import cv2
from functools import lru_cache
from typing import Iterable


def _l2(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    n = float(np.linalg.norm(v))
    return v / max(n, eps)


class OpenCLIPBackend:
    def __init__(self, input_clip_model_name: str, input_clip_pretrained: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model, _, preproc = open_clip.create_model_and_transforms(
            model_name=input_clip_model_name, pretrained=input_clip_pretrained
        )
        self.model = model.to(self.device).eval()
        self.preproc = preproc

    @torch.inference_mode()
    def embed_rgb_np(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """RGB uint8 [H,W,3] -> L2-normalized embedding [D]"""
        pil = Image.fromarray(rgb_uint8)
        x = self.preproc(pil).unsqueeze(0).to(self.device)
        v = self.model.encode_image(
            x)[0].detach().cpu().numpy().astype(np.float32)
        return _l2(v)

    def embed_multiscale(self, bgr_uint8: np.ndarray, scales: Iterable[float]) -> list[np.ndarray]:
        if bgr_uint8 is None:
            raise ValueError("embed_multiscale: input image is None")
        # rgb0 = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)
        rgb0 = bgr_uint8
        H, W = rgb0.shape[:2]
        out: list[np.ndarray] = []
        for s in scales:
            w = max(1, int(W * s))
            h = max(1, int(H * s))
            rgb = cv2.resize(rgb0, (w, h), interpolation=cv2.INTER_AREA)
            out.append(self.embed_rgb_np(rgb))
        return out


@lru_cache(maxsize=1)
def get_openclip_backend(input_clip_model_name: str, input_clip_pretrained: str) -> OpenCLIPBackend:
    """โหลด backend ครั้งเดียวต่อโปรเซส"""
    return OpenCLIPBackend(input_clip_model_name=input_clip_model_name, input_clip_pretrained=input_clip_pretrained)
