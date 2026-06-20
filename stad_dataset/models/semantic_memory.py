"""
CLIP + ChromaDB semantic memory for object-centric anomaly scoring.

This module stores embeddings from previously observed objects and
scores current objects by nearest-neighbor semantic distance.
"""

import logging
import os
import uuid
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ClipChromaMemory:
    """Persistent semantic memory using CLIP image embeddings and ChromaDB."""

    def __init__(self, config):
        self.config = config
        self.enabled = bool(getattr(config, "enabled", True))
        self._model = None
        self._processor = None
        self._torch = None
        self._collection = None
        self._device = getattr(config, "device", "cpu")
        self._top_k = max(1, int(getattr(config, "top_k", 5)))
        self._min_history = max(1, int(getattr(config, "min_history", 25)))
        self._min_crop = max(8, int(getattr(config, "min_crop_size", 24)))

        if not self.enabled:
            logger.info("Semantic memory disabled by config")
            return

        try:
            import chromadb
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor

            self._torch = torch
            self._image_cls = Image

            self._model = CLIPModel.from_pretrained(config.clip_model_name)
            self._processor = CLIPProcessor.from_pretrained(config.clip_model_name)
            self._model.to(self._device)
            self._model.eval()

            persist_dir = getattr(config, "chroma_persist_dir", "outputs/chroma")
            os.makedirs(persist_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=persist_dir)

            # Use a run-unique collection to avoid cross-run leakage.
            run_name = f"{config.collection_name}_{uuid.uuid4().hex[:8]}"
            self._collection = client.get_or_create_collection(
                name=run_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Semantic memory initialized: CLIP + ChromaDB")
        except Exception as exc:
            self.enabled = False
            logger.warning(f"Semantic memory disabled (init failed): {exc}")

    def _extract_crop(self, image: np.ndarray, track: Dict) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        cx = int(track.get("cx", 0))
        cy = int(track.get("cy", 0))
        bw = int(max(track.get("width", 0), self._min_crop))
        bh = int(max(track.get("height", 0), self._min_crop))

        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w, cx + bw // 2)
        y2 = min(h, cy + bh // 2)

        if x2 <= x1 or y2 <= y1:
            return None
        crop = image[y1:y2, x1:x2]
        if crop.shape[0] < self._min_crop or crop.shape[1] < self._min_crop:
            return None
        return crop

    def _embed_crop(self, crop_bgr: np.ndarray) -> Optional[List[float]]:
        if not self.enabled:
            return None

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = self._image_cls.fromarray(rgb)
        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with self._torch.no_grad():
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        return feats[0].detach().cpu().numpy().astype(np.float32).tolist()

    def count(self) -> int:
        if not self.enabled:
            return 0
        try:
            return int(self._collection.count())
        except Exception:
            return 0

    def ingest_tracks(self, frame_bgr: np.ndarray, tracks: List[Dict], frame_id: int) -> int:
        """Store embeddings from tracks as normal-history memory entries."""
        if not self.enabled or not tracks:
            return 0

        added = 0
        for tr in tracks:
            tid = int(tr.get("track_id", -1))
            if tid < 0:
                continue

            crop = self._extract_crop(frame_bgr, tr)
            if crop is None:
                continue

            emb = self._embed_crop(crop)
            if emb is None:
                continue

            cls = str(tr.get("class_name", "unknown"))
            entry_id = f"f{frame_id}_t{tid}_{added}"
            meta = {
                "class_name": cls,
                "frame_id": int(frame_id),
                "track_id": tid,
            }
            try:
                self._collection.add(
                    ids=[entry_id],
                    embeddings=[emb],
                    metadatas=[meta],
                )
                added += 1
            except Exception:
                continue

        return added

    def score_tracks(self, frame_bgr: np.ndarray, tracks: List[Dict]) -> Dict[int, float]:
        """
        Return semantic anomaly score per track in [0,1].
        Higher score means stronger semantic deviation from memory.
        """
        scores: Dict[int, float] = {}
        if not self.enabled or not tracks:
            return scores

        if self.count() < self._min_history:
            for tr in tracks:
                tid = int(tr.get("track_id", -1))
                if tid >= 0:
                    scores[tid] = 0.5
            return scores

        for tr in tracks:
            tid = int(tr.get("track_id", -1))
            if tid < 0:
                continue

            crop = self._extract_crop(frame_bgr, tr)
            if crop is None:
                scores[tid] = 0.5
                continue

            emb = self._embed_crop(crop)
            if emb is None:
                scores[tid] = 0.5
                continue

            cls = str(tr.get("class_name", "unknown"))
            dist = None

            # Prefer class-conditional memory lookup first.
            try:
                q = self._collection.query(
                    query_embeddings=[emb],
                    n_results=self._top_k,
                    where={"class_name": cls},
                    include=["distances"],
                )
                if q and q.get("distances") and q["distances"][0]:
                    dist = float(min(q["distances"][0]))
            except Exception:
                dist = None

            # Fallback to global memory.
            if dist is None:
                try:
                    q = self._collection.query(
                        query_embeddings=[emb],
                        n_results=self._top_k,
                        include=["distances"],
                    )
                    if q and q.get("distances") and q["distances"][0]:
                        dist = float(min(q["distances"][0]))
                except Exception:
                    dist = None

            if dist is None:
                scores[tid] = 0.5
                continue

            # Chroma cosine distance is typically in [0, 2].
            anomaly_score = float(np.clip(dist / 2.0, 0.0, 1.0))
            scores[tid] = anomaly_score

        return scores
