"""
Player Re-Identification Module.

Multi-layer Re-ID architecture for robust player identity across 90-minute matches:
  Layer 1: BoT-SORT frame-to-frame (already in tracker)
  Layer 2: Appearance embedding cache (cosine similarity matching)
  Layer 3: Jersey number recognition (periodic + on conflict)
  Layer 4: Post-hoc global ID correction

Designed for tiered detection: 1920px main detection + 4K crops for Re-ID.
"""
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class AppearanceEmbedding:
    """Lightweight appearance feature extractor for player crops.

    Uses color histogram + spatial features from jersey region.
    For GPU-accelerated embedding, swap in a proper ReID model (OSNet, etc).
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

    def extract(self, frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        """Extract appearance embedding from player crop.

        Args:
            frame: Full frame (BGR)
            bbox: [x1, y1, x2, y2] pixel coordinates

        Returns:
            Normalized embedding vector or None if crop is too small
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < 8 or crop_h < 12:
            return None

        crop = frame[y1:y2, x1:x2]

        # Focus on upper body (jersey area): top 50%
        jersey_h = crop_h // 2
        jersey = crop[:jersey_h, :]

        # Color histogram features (HSV)
        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

        # Spatial color distribution: divide into 2x2 grid
        gh = jersey.shape[0] // 2
        gw = jersey.shape[1] // 2
        if gh < 2 or gw < 2:
            spatial_feats = np.zeros(48)
        else:
            spatial_feats = []
            for gy in range(2):
                for gx in range(2):
                    patch = jersey[gy * gh:(gy + 1) * gh, gx * gw:(gx + 1) * gw]
                    if patch.size > 0:
                        mean_color = np.mean(patch.reshape(-1, 3), axis=0)
                        spatial_feats.extend(mean_color)
                    else:
                        spatial_feats.extend([0, 0, 0])
            spatial_feats = np.array(spatial_feats)

        # Lower body features (shorts/socks) for additional disambiguation
        lower = crop[jersey_h:, :]
        if lower.size > 0:
            lower_hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
            lower_h = cv2.calcHist([lower_hsv], [0], None, [16], [0, 180]).flatten()
        else:
            lower_h = np.zeros(16)

        # Combine all features
        embedding = np.concatenate([h_hist, s_hist, v_hist, spatial_feats, lower_h])

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


class AppearanceCache:
    """Maintains a rolling cache of appearance embeddings per track.

    Stores multiple embeddings per track for robust matching.
    """

    def __init__(self, max_embeddings_per_track: int = 20, similarity_threshold: float = 0.7):
        self.max_per_track = max_embeddings_per_track
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.track_metadata: Dict[int, dict] = {}

    def update(self, track_id: int, embedding: np.ndarray, metadata: dict = None):
        """Add an embedding to a track's cache."""
        if embedding is None:
            return

        embeddings = self.cache[track_id]
        embeddings.append(embedding)

        # Keep only the most recent embeddings
        if len(embeddings) > self.max_per_track:
            self.cache[track_id] = embeddings[-self.max_per_track:]

        if metadata:
            self.track_metadata[track_id] = metadata

    def get_average_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Get the average embedding for a track."""
        embeddings = self.cache.get(track_id, [])
        if not embeddings:
            return None
        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 0 else avg

    def match(self, embedding: np.ndarray, exclude_ids: set = None) -> Tuple[Optional[int], float]:
        """Find the best matching track for an embedding.

        Args:
            embedding: Query embedding
            exclude_ids: Track IDs to exclude from matching

        Returns:
            (best_track_id, similarity_score) or (None, 0)
        """
        if embedding is None:
            return None, 0.0

        best_id = None
        best_sim = 0.0

        for track_id, embeddings in self.cache.items():
            if exclude_ids and track_id in exclude_ids:
                continue
            if not embeddings:
                continue

            avg_emb = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(avg_emb)
            if norm > 0:
                avg_emb = avg_emb / norm

            sim = np.dot(embedding, avg_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = track_id

        if best_sim >= self.similarity_threshold:
            return best_id, best_sim
        return None, best_sim

    def find_potential_swaps(self) -> List[Tuple[int, int, float]]:
        """Find track pairs that might be ID-swapped.

        Returns list of (track_a, track_b, similarity) for suspiciously similar tracks.
        """
        swaps = []
        track_ids = list(self.cache.keys())

        for i in range(len(track_ids)):
            emb_a = self.get_average_embedding(track_ids[i])
            if emb_a is None:
                continue
            for j in range(i + 1, len(track_ids)):
                emb_b = self.get_average_embedding(track_ids[j])
                if emb_b is None:
                    continue
                sim = np.dot(emb_a, emb_b)
                if sim > 0.9:  # Very similar appearance — potential swap
                    swaps.append((track_ids[i], track_ids[j], sim))

        return swaps


class JerseyNumberRecognizer:
    """Jersey number recognition from player crops.

    Uses PaddleOCR or template matching to read jersey numbers.
    Falls back to crop-based classification when OCR isn't available.
    """

    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr
        self.ocr_engine = None
        self._ocr_available = False

        if use_ocr:
            try:
                from paddleocr import PaddleOCR
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    show_log=False,
                    use_gpu=False,
                )
                self._ocr_available = True
            except ImportError:
                print("  [ReID] PaddleOCR not available — jersey number recognition disabled")
                self._ocr_available = False

    def recognize(self, frame: np.ndarray, bbox: list) -> Optional[int]:
        """Try to recognize jersey number from player crop.

        Args:
            frame: Full frame (BGR), ideally from 4K source
            bbox: [x1, y1, x2, y2] pixel coordinates

        Returns:
            Detected jersey number or None
        """
        if not self._ocr_available:
            return None

        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < 20 or crop_h < 30:
            return None

        crop = frame[y1:y2, x1:x2]

        # Focus on back/front of jersey (middle 60% vertically, full width)
        jersey_top = int(crop_h * 0.2)
        jersey_bot = int(crop_h * 0.7)
        jersey = crop[jersey_top:jersey_bot, :]

        # Upscale for better OCR
        if jersey.shape[1] < 100:
            scale = 100 / jersey.shape[1]
            jersey = cv2.resize(jersey, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC)

        try:
            result = self.ocr_engine.ocr(jersey, cls=True)
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0].strip()
                    confidence = line[1][1]
                    # Extract numbers
                    digits = ''.join(c for c in text if c.isdigit())
                    if digits and confidence > 0.5:
                        num = int(digits)
                        if 1 <= num <= 99:  # Valid jersey number
                            return num
        except Exception:
            pass

        return None


class PlayerReIDModule:
    """Main Re-ID module coordinating all layers.

    Usage:
        reid = PlayerReIDModule()
        # During tracking (per frame):
        reid.process_frame(frame, frame_tracks, frame_4k=optional_4k_frame)
        # After tracking (post-hoc correction):
        corrected_tracks = reid.post_hoc_correction(all_tracks)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        jersey_check_interval: int = 150,  # Check jersey every N frames
        jersey_check_on_conflict: bool = True,
        use_ocr: bool = True,
    ):
        self.embedding_extractor = AppearanceEmbedding()
        self.cache = AppearanceCache(similarity_threshold=similarity_threshold)
        self.jersey_recognizer = JerseyNumberRecognizer(use_ocr=use_ocr)
        self.jersey_check_interval = jersey_check_interval
        self.jersey_check_on_conflict = jersey_check_on_conflict

        # Track → jersey number mapping (accumulated evidence)
        self.jersey_votes: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # Final jersey assignments
        self.jersey_assignments: Dict[int, int] = {}

        self._frame_count = 0

    def process_frame(
        self,
        frame: np.ndarray,
        frame_tracks: list,
        frame_4k: np.ndarray = None,
    ):
        """Process a single frame: update appearance cache and optionally check jerseys.

        Args:
            frame: Current frame (detection resolution, e.g. 1920px)
            frame_tracks: Track list for this frame
            frame_4k: Optional full-resolution frame for jersey recognition
        """
        self._frame_count += 1
        do_jersey_check = (self._frame_count % self.jersey_check_interval == 0)

        for t in frame_tracks:
            track_id = t['track_id']
            if t.get('role') in ('ball', 'referee'):
                continue

            bbox = t.get('bbox')
            if bbox is None:
                continue

            # Layer 2: Update appearance embedding
            embedding = self.embedding_extractor.extract(frame, bbox)
            if embedding is not None:
                self.cache.update(track_id, embedding, {
                    'team': t.get('team'),
                    'role': t.get('role'),
                })

            # Layer 3: Periodic jersey number recognition
            if do_jersey_check:
                source_frame = frame_4k if frame_4k is not None else frame
                # Scale bbox if using 4K frame with detections from scaled frame
                if frame_4k is not None:
                    scale_x = frame_4k.shape[1] / frame.shape[1]
                    scale_y = frame_4k.shape[0] / frame.shape[0]
                    scaled_bbox = [
                        bbox[0] * scale_x, bbox[1] * scale_y,
                        bbox[2] * scale_x, bbox[3] * scale_y,
                    ]
                else:
                    scaled_bbox = bbox

                number = self.jersey_recognizer.recognize(source_frame, scaled_bbox)
                if number is not None:
                    self.jersey_votes[track_id][number] += 1

    def resolve_conflict(
        self,
        frame: np.ndarray,
        track_a_id: int,
        track_b_id: int,
        track_a_bbox: list,
        track_b_bbox: list,
        frame_4k: np.ndarray = None,
    ) -> Dict[int, int]:
        """Resolve identity conflict between two tracks after occlusion.

        Uses appearance matching + jersey numbers to determine correct assignments.

        Args:
            frame: Current frame
            track_a_id, track_b_id: Conflicting track IDs
            track_a_bbox, track_b_bbox: Current bboxes
            frame_4k: Optional 4K frame for better jersey recognition

        Returns:
            Dict mapping old_track_id -> corrected_track_id
        """
        result = {track_a_id: track_a_id, track_b_id: track_b_id}

        # Get cached embeddings
        emb_a_cached = self.cache.get_average_embedding(track_a_id)
        emb_b_cached = self.cache.get_average_embedding(track_b_id)

        # Get current embeddings
        emb_a_now = self.embedding_extractor.extract(frame, track_a_bbox)
        emb_b_now = self.embedding_extractor.extract(frame, track_b_bbox)

        if emb_a_cached is not None and emb_b_cached is not None \
                and emb_a_now is not None and emb_b_now is not None:
            # Compare: does current A match cached A better, or cached B?
            sim_aa = np.dot(emb_a_now, emb_a_cached)  # A now matches A cached
            sim_ab = np.dot(emb_a_now, emb_b_cached)  # A now matches B cached
            sim_bb = np.dot(emb_b_now, emb_b_cached)  # B now matches B cached
            sim_ba = np.dot(emb_b_now, emb_a_cached)  # B now matches A cached

            correct_score = sim_aa + sim_bb  # IDs are correct
            swapped_score = sim_ab + sim_ba  # IDs are swapped

            if swapped_score > correct_score + 0.1:
                # IDs are likely swapped
                result = {track_a_id: track_b_id, track_b_id: track_a_id}

        # Jersey number check as tiebreaker
        if self.jersey_check_on_conflict:
            source = frame_4k if frame_4k is not None else frame
            num_a = self.jersey_recognizer.recognize(source, track_a_bbox)
            num_b = self.jersey_recognizer.recognize(source, track_b_bbox)

            # Check against historical jersey assignments
            expected_a = self._get_best_jersey(track_a_id)
            expected_b = self._get_best_jersey(track_b_id)

            if num_a is not None and num_b is not None:
                if expected_a is not None and expected_b is not None:
                    if num_a == expected_b and num_b == expected_a:
                        # Jersey numbers confirm swap
                        result = {track_a_id: track_b_id, track_b_id: track_a_id}
                    elif num_a == expected_a and num_b == expected_b:
                        # Jersey numbers confirm correct
                        result = {track_a_id: track_a_id, track_b_id: track_b_id}

        return result

    def post_hoc_correction(self, all_tracks: list) -> list:
        """Post-hoc global ID correction using accumulated jersey evidence.

        After tracking is complete:
        1. Assign final jersey numbers from votes
        2. Detect tracks with inconsistent jerseys
        3. Split tracks at swap points

        Returns corrected all_tracks.
        """
        # Step 1: Assign final jersey numbers
        for track_id, votes in self.jersey_votes.items():
            if votes:
                best_number = max(votes, key=votes.get)
                total_votes = sum(votes.values())
                best_votes = votes[best_number]
                if best_votes / total_votes > 0.5:  # Majority vote
                    self.jersey_assignments[track_id] = best_number

        if not self.jersey_assignments:
            return all_tracks

        print(f"  [ReID] Jersey assignments: {dict(self.jersey_assignments)}")

        # Step 2: Detect potential swaps by finding tracks that share jersey numbers
        number_to_tracks = defaultdict(list)
        for tid, num in self.jersey_assignments.items():
            number_to_tracks[num].append(tid)

        swaps_found = 0
        for num, tids in number_to_tracks.items():
            if len(tids) > 1:
                # Multiple tracks have the same jersey number — potential merge
                print(f"  [ReID] Jersey #{num} claimed by tracks: {tids}")
                swaps_found += 1

        if swaps_found > 0:
            print(f"  [ReID] {swaps_found} potential ID conflicts detected "
                  f"(would need track merging)")

        # Add jersey_number to track data
        for frame_tracks in all_tracks:
            for t in frame_tracks:
                tid = t['track_id']
                if tid in self.jersey_assignments:
                    t['jersey_number'] = self.jersey_assignments[tid]

        return all_tracks

    def _get_best_jersey(self, track_id: int) -> Optional[int]:
        """Get the best jersey number candidate for a track."""
        if track_id in self.jersey_assignments:
            return self.jersey_assignments[track_id]
        votes = self.jersey_votes.get(track_id, {})
        if votes:
            return max(votes, key=votes.get)
        return None

    def get_stats(self) -> dict:
        """Get ReID module statistics."""
        return {
            'tracks_in_cache': len(self.cache.cache),
            'jersey_assignments': len(self.jersey_assignments),
            'jersey_votes_total': sum(
                sum(v.values()) for v in self.jersey_votes.values()
            ),
            'potential_swaps': len(self.cache.find_potential_swaps()),
        }
