import cv2
import numpy as np
from loguru import logger
from numpy.linalg import norm
from config.settings import FACE_MATCH_THRESHOLD, FACE_MIN_QUALITY, FACE_MIN_SIZE

try:
    from insightface.app import FaceAnalysis
except ImportError:
    pass


class RecognitionAgent:
    """
    Detects faces in an RGB frame and matches them against the dataset.

    Two-stage matching pipeline:
    ────────────────────────────
    Stage 1 — Centroid match
        Compare the detected embedding against each person's averaged centroid.
        Fast. Accurate when the dataset has multiple photos per person.

    Stage 2 — Top-K vote  (borderline cases only)
        If the centroid score falls within VOTE_MARGIN of the threshold,
        compare against every individual stored embedding and use majority
        voting to make the final call. Prevents borderline scores from
        producing wrong labels.

    Quality gate:
        Faces below FACE_MIN_QUALITY detection score or smaller than
        FACE_MIN_SIZE pixels are ignored — this eliminates blurry /
        partial / background faces that would hurt accuracy.
    """

    VOTE_MARGIN  = 0.05   # trigger voting when score ∈ [threshold-margin, threshold)
    TOP_K        = 15     # number of top individual-embedding matches used for voting

    def __init__(self, dataset_agent):
        self.dataset = dataset_agent

        try:
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception as e:
            logger.error(f"Failed to initialise InsightFace: {e}")
            self.app = None

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = norm(a), norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _l2_norm(v: np.ndarray) -> np.ndarray:
        n = norm(v)
        return v / n if n > 0 else v

    def _centroid_match(self, emb: np.ndarray):
        """Return (best_name, best_score) from centroid comparison."""
        best_score = -1.0
        best_name  = "Unknown"
        for centroid, name in zip(
            self.dataset.known_face_encodings,
            self.dataset.known_face_names,
        ):
            s = self._cosine(centroid, emb)
            if s > best_score:
                best_score, best_name = s, name
        return best_name, best_score

    def _vote_match(self, emb: np.ndarray, candidate: str) -> bool:
        """
        Top-K majority vote across all individual embeddings.
        Returns True if the candidate wins the vote.
        """
        all_embeds = getattr(self.dataset, "known_face_all", {})
        if not all_embeds:
            return False

        scores = []
        for name, vecs in all_embeds.items():
            for vec in vecs:
                scores.append((self._cosine(vec, emb), name))

        scores.sort(reverse=True)
        top = scores[: self.TOP_K]

        vote: dict[str, int] = {}
        for sc, nm in top:
            if sc >= FACE_MATCH_THRESHOLD:
                vote[nm] = vote.get(nm, 0) + 1

        if not vote:
            return False

        winner = max(vote, key=vote.get)
        return winner == candidate

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def process_frame(self, frame_rgb: np.ndarray) -> list:
        """
        Detect and identify all faces in an RGB frame.

        Returns:
            list[dict] — keys: name, confidence, box (top, right, bottom, left)
        """
        results = []
        if not self.app:
            return results

        try:
            faces = self.app.get(frame_rgb)
        except Exception as e:
            logger.error(f"InsightFace inference error: {e}")
            return results

        if not faces:
            return results

        for face in faces:

            # ── Quality + size gate ──────────────────────────────────
            det_score = float(getattr(face, "det_score", 1.0))
            box       = face.bbox.astype(int)
            left, top, right, bottom = box[0], box[1], box[2], box[3]
            side = min(right - left, bottom - top)

            if det_score < FACE_MIN_QUALITY or side < FACE_MIN_SIZE:
                continue   # skip low-quality / tiny face

            # ── Embedding ────────────────────────────────────────────
            emb = self._l2_norm(face.normed_embedding)

            name       = "Unknown"
            confidence = 0.0

            if self.dataset.known_face_encodings:

                candidate, score = self._centroid_match(emb)

                if score >= FACE_MATCH_THRESHOLD:
                    # ✅ Clear centroid match
                    name       = candidate
                    confidence = min(score, 1.0)

                elif score >= FACE_MATCH_THRESHOLD - self.VOTE_MARGIN:
                    # ⚖️ Borderline — use voting to decide
                    if self._vote_match(emb, candidate):
                        name       = candidate
                        confidence = min(score, 1.0)
                    # else: remains Unknown

            results.append({
                "name":       name,
                "confidence": round(confidence, 4),
                "box":        (int(top), int(right), int(bottom), int(left)),
            })

        return results
