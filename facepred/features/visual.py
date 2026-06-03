"""Visual feature extraction using MediaPipe Face Landmarker.

Extracts face landmarks (478 × 3), blendshape coefficients (52),
head pose (6-DoF), and confidence from RGB video frames.
All processing runs on CPU at 30+ FPS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# MediaPipe landmark counts and indices
NUM_LANDMARKS = 478
LANDMARK_DIM = NUM_LANDMARKS * 3  # x, y, z per landmark
NUM_BLENDSHAPES = 52
HEAD_POSE_DIM = 6  # rotation (3) + translation (3)
CONFIDENCE_DIM = 1

# Mouth-specific landmark indices (for mouth ROI extraction)
# These are the MediaPipe face mesh mouth landmarks
MOUTH_LANDMARK_INDICES = list(range(61, 81)) + list(range(81, 96)) + [
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
    146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312,
    314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
]

# Total visual feature dimension
TOTAL_VISUAL_DIM = LANDMARK_DIM + NUM_BLENDSHAPES + HEAD_POSE_DIM + CONFIDENCE_DIM  # 1493


@dataclass
class VisualFeatures:
    """Container for extracted visual features from a single frame.

    Attributes:
        landmarks: Face landmark positions [478, 3] (x, y, z normalized).
        blendshapes: Blendshape coefficients [52] (0-1 range).
        head_pose: Head rotation and translation [6] (radians + normalized).
        confidence: Detection confidence [1] (0-1).
        timestamp_ms: Frame timestamp in milliseconds.
    """

    landmarks: torch.Tensor  # [478, 3]
    blendshapes: torch.Tensor  # [52]
    head_pose: torch.Tensor  # [6]
    confidence: torch.Tensor  # [1]
    timestamp_ms: float = 0.0

    def to_flat_tensor(self) -> torch.Tensor:
        """Flatten all features into a single [1493]-dim vector."""
        return torch.cat([
            self.landmarks.flatten(),  # [1434]
            self.blendshapes,          # [52]
            self.head_pose,            # [6]
            self.confidence,           # [1]
        ])

    @staticmethod
    def zeros() -> VisualFeatures:
        """Return zero features (for missing/failed detection)."""
        return VisualFeatures(
            landmarks=torch.zeros(NUM_LANDMARKS, 3),
            blendshapes=torch.zeros(NUM_BLENDSHAPES),
            head_pose=torch.zeros(HEAD_POSE_DIM),
            confidence=torch.zeros(CONFIDENCE_DIM),
        )


class VisualExtractor:
    """Extract face features from RGB frames using MediaPipe.

    Runs entirely on CPU. Supports both single-frame and batch extraction.

    Args:
        min_detection_confidence: Minimum confidence for face detection (0-1).
        min_tracking_confidence: Minimum confidence for landmark tracking (0-1).
        max_num_faces: Maximum number of faces to detect.

    Example:
        >>> extractor = VisualExtractor()
        >>> frame = cv2.imread('face.jpg')
        >>> features = extractor.extract_frame(frame)
        >>> features.to_flat_tensor().shape  # [1493]
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 1,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_faces = max_num_faces
        self._landmarker = None
        self._is_initialized = False

    def _init_landmarker(self) -> None:
        """Lazy-initialize MediaPipe Face Landmarker."""
        if self._is_initialized:
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            # Download model if needed
            model_path = self._get_model_path()

            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=self.max_num_faces,
                min_face_detection_confidence=self.min_detection_confidence,
                min_face_presence_confidence=self.min_tracking_confidence,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._mp_image_class = mp.Image
            self._is_initialized = True
            logger.info("MediaPipe Face Landmarker initialized successfully")

        except ImportError:
            raise ImportError(
                "mediapipe is required for visual feature extraction. "
                "Install with: pip install mediapipe"
            )

    def _get_model_path(self) -> Path:
        """Get path to MediaPipe face landmarker model, downloading if needed."""
        import urllib.request

        model_dir = Path.home() / ".facepred" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "face_landmarker_v2_with_blendshapes.task"

        if not model_path.exists():
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/"
                "face_landmarker.task"
            )
            logger.info(f"Downloading MediaPipe face landmarker model to {model_path}")
            urllib.request.urlretrieve(url, model_path)
            logger.info("Download complete")

        return model_path

    def extract_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float = 0.0,
    ) -> VisualFeatures:
        """Extract visual features from a single RGB frame.

        Args:
            frame: RGB image as numpy array, shape [H, W, 3], dtype uint8.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            VisualFeatures dataclass with all extracted features.
            Returns zeros if no face is detected.
        """
        self._init_landmarker()

        # Convert to MediaPipe image
        mp_image = self._mp_image_class(
            image_format=self._mp_image_class.ImageFormat.SRGB,
            data=frame,
        )

        # Detect landmarks
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            logger.debug(f"No face detected at t={timestamp_ms:.0f}ms")
            return VisualFeatures.zeros()

        # Use first detected face
        face = result.face_landmarks[0]

        # Extract landmarks [478, 3]
        landmarks = torch.tensor(
            [[lm.x, lm.y, lm.z] for lm in face],
            dtype=torch.float32,
        )

        # Extract blendshapes [52]
        blendshapes = torch.zeros(NUM_BLENDSHAPES, dtype=torch.float32)
        if result.face_blendshapes:
            bs = result.face_blendshapes[0]
            for i, category in enumerate(bs):
                if i < NUM_BLENDSHAPES:
                    blendshapes[i] = category.score

        # Extract head pose [6] from transformation matrix
        head_pose = torch.zeros(HEAD_POSE_DIM, dtype=torch.float32)
        if result.facial_transformation_matrixes:
            matrix = result.facial_transformation_matrixes[0]
            mat = np.array(matrix).reshape(4, 4)
            # Extract rotation (simplified: use Euler-like from rotation matrix)
            head_pose[:3] = torch.tensor([mat[0, 3], mat[1, 3], mat[2, 3]])
            # Translation
            head_pose[3:] = torch.tensor([mat[0, 0], mat[1, 1], mat[2, 2]])

        # Confidence
        confidence = torch.tensor([1.0], dtype=torch.float32)  # Face was detected

        return VisualFeatures(
            landmarks=landmarks,
            blendshapes=blendshapes,
            head_pose=head_pose,
            confidence=confidence,
            timestamp_ms=timestamp_ms,
        )

    def extract_video(
        self,
        video_path: Union[str, Path],
        fps: int = 30,
        max_frames: Optional[int] = None,
    ) -> list[VisualFeatures]:
        """Extract visual features from all frames of a video file.

        Args:
            video_path: Path to video file.
            fps: Target FPS for extraction (will skip frames if video FPS is higher).
            max_frames: Maximum number of frames to process (None = all).

        Returns:
            List of VisualFeatures, one per extracted frame.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python is required. Install with: pip install opencv-python")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(video_fps / fps))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        features_list = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Convert BGR (OpenCV) to RGB (MediaPipe)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = (frame_idx / video_fps) * 1000.0

                features = self.extract_frame(rgb_frame, timestamp_ms)
                features_list.append(features)

                if max_frames and len(features_list) >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        logger.info(
            f"Extracted {len(features_list)} frames from {video_path} "
            f"({total_frames} total frames, skip={frame_skip})"
        )

        return features_list

    def get_feature_dim(self) -> int:
        """Return total visual feature dimension."""
        return TOTAL_VISUAL_DIM

    def get_zeros(self, num_frames: int = 1) -> torch.Tensor:
        """Return zero feature tensor for missing data. Shape: [num_frames, 1493]."""
        return torch.zeros(num_frames, TOTAL_VISUAL_DIM, dtype=torch.float32)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._is_initialized = False
