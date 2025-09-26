import os
import cv2
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.feature import hog
from typing import List, Dict, Tuple, Literal

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageEncoder:
    """Interface for encoding image data into vector representation."""
    def __init__(self, encoder_type: Literal["HOG", "SIFT", "Flatten"]) -> None:
        self.encoder_type = encoder_type

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Method for encoding image data."""
        if self.encoder_type == "HOG":
            return self.hog_encoder(image)
        elif self.encoder_type == "SIFT":
            return self.sift_encoder(image)
        elif self.encoder_type == "Flatten":
            return self.flatten_encoder(image)

    def hog_encoder(self, image: np.ndarray) -> np.ndarray:
        """Method for encoding image data by using HOG (Histogram of Oriented Gradients) approach."""
        hog_features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            block_norm='L2-Hys',
            feature_vector=True,
            channel_axis=None
        )

        logger.debug(f"HOG features shape: {hog_features.shape}")
        return hog_features

    def sift_encoder(self, image: np.ndarray) -> np.ndarray:
        """Method for encoding image data by using SIFT (Scale-Invariant Feature Transform)."""
        sift = cv2.SIFT_create()

        keypoints, descriptors = sift.detectAndCompute(image, None)

        if descriptors is None:
            logger.warning("No SIFT features found in the image, returning zero vector")
            return np.zeros(128, dtype=np.float32)  # SIFT descriptor size = 128

        # Aggregate descriptors (here: average pooling)
        sift_vector = np.mean(descriptors, axis=0)

        logger.debug(f"SIFT features shape: {sift_vector.shape}")
        return sift_vector.astype(np.float32)

    def flatten_encoder(self, image: np.ndarray) -> np.ndarray:
        """Method for encoding image data by flattening pixels into a single 1D vector."""
        return image.flatten()
