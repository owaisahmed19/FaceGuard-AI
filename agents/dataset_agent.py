import os
import sys
import cv2
import pickle
import numpy as np
from pathlib import Path
from loguru import logger

try:
    from insightface.app import FaceAnalysis
except ImportError:
    logger.warning("insightface not installed. Run `pip install insightface==0.7.3 onnxruntime`")

sys.path.insert(0, ".")
from config.settings import DATASET_DIR, EMBEDDINGS_FILE, FACE_MIN_QUALITY, FACE_MIN_SIZE


class DatasetAgent:
    """
    Builds a high-quality face embedding database.

    Accuracy techniques used:
    ─────────────────────────
    1. Per-person CENTROID  — all photos of a person are averaged into one
                              representative vector (more robust than single photos).
    2. Flip augmentation    — each image is also processed horizontally flipped,
                              effectively doubling the training samples for free.
    3. Quality filtering    — faces below InsightFace det_score threshold are
                              discarded (blurry / occluded / side profiles).
    4. Size filtering       — tiny faces (< FACE_MIN_SIZE px) are rejected.
    5. L2 re-normalisation  — the averaged centroid is re-normalised so cosine
                              similarity remains in the correct [0,1] range.
    """

    def __init__(self):
        self.known_face_encodings = []   # one L2-normalised centroid per person
        self.known_face_names     = []
        self.known_face_all       = {}   # {name: [all individual embeddings]} — used for voting

        try:
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception as e:
            logger.error(f"Failed to initialise InsightFace: {e}")
            self.app = None

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _l2_norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _extract_embedding(self, img_rgb: np.ndarray):
        """
        Run InsightFace on a single RGB image.
        Applies quality + size gate and returns the normalised embedding
        of the largest qualifying face, or None.
        """
        try:
            faces = self.app.get(img_rgb)
        except Exception:
            return None

        if not faces:
            return None

        good = []
        for f in faces:
            score = float(getattr(f, "det_score", 1.0))
            box   = f.bbox.astype(int)
            side  = min(box[2] - box[0], box[3] - box[1])
            if score >= FACE_MIN_QUALITY and side >= FACE_MIN_SIZE:
                good.append((f, side))

        if not good:
            return None

        # Largest passing face
        best_face = max(good, key=lambda x: x[1])[0]
        return self._l2_norm(best_face.normed_embedding)

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def load_embeddings(self) -> bool:
        """Load cached embeddings from disk if available."""
        if not EMBEDDINGS_FILE.exists():
            return False
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                data = pickle.load(f)
            self.known_face_encodings = data.get("encodings", [])
            self.known_face_names     = data.get("names", [])
            self.known_face_all       = data.get("all_embeddings", {})
            logger.info(f"Loaded {len(self.known_face_names)} identity centroids from cache.")
            return True
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False

    def rebuild_embeddings(self) -> bool:
        """
        Scan DATASET_DIR and rebuild all embeddings from scratch.

        Folder layout expected:
            people_dataset/
                Person_Name/
                    photo1.jpg
                    photo2.jpg
                    ...
        """
        if not self.app:
            logger.error("InsightFace model not initialised.")
            return False

        if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
            logger.warning("Dataset directory is empty. Add person folders first.")
            return False

        centroids  = []
        names      = []
        all_embeds = {}

        for person_dir in sorted(DATASET_DIR.iterdir()):
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            vecs        = []
            n_images    = 0

            logger.info(f"  Processing identity: {person_name}")

            for img_file in person_dir.iterdir():
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue

                raw = cv2.imread(str(img_file))
                if raw is None:
                    continue
                n_images += 1

                img_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

                # ① Original image
                emb = self._extract_embedding(img_rgb)
                if emb is not None:
                    vecs.append(emb)

                # ② Horizontal flip  (free augmentation — doubles useful data)
                flipped = cv2.flip(img_rgb, 1)
                emb_f   = self._extract_embedding(flipped)
                if emb_f is not None:
                    vecs.append(emb_f)

                # ③ Slight brightness boost  (handles dim photos)
                bright   = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=15)
                emb_b    = self._extract_embedding(bright)
                if emb_b is not None:
                    vecs.append(emb_b)

            if not vecs:
                logger.warning(f"    No valid faces found for '{person_name}' — skipping.")
                continue

            # ④ Average all embeddings → centroid → re-normalise
            centroid = self._l2_norm(np.mean(vecs, axis=0))

            centroids.append(centroid)
            names.append(person_name)
            all_embeds[person_name] = vecs

            logger.info(
                f"    {n_images} images  →  {len(vecs)} embeddings  →  1 centroid"
            )

        if not centroids:
            logger.error("Rebuild failed — no valid embeddings produced.")
            return False

        self.known_face_encodings = centroids
        self.known_face_names     = names
        self.known_face_all       = all_embeds

        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump({
                "encodings":      centroids,
                "names":          names,
                "all_embeddings": all_embeds,
            }, f)

        total_vecs = sum(len(v) for v in all_embeds.values())
        logger.info(
            f"Rebuild complete — {len(names)} identities, "
            f"{total_vecs} total embeddings, 1 centroid per person."
        )
        return True


if __name__ == "__main__":
    agent = DatasetAgent()
    agent.rebuild_embeddings()
