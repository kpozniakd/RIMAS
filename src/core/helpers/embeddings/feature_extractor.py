import os
import sys
import cv2
import json
import string
import logging
import numpy as np
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
from skimage.feature import hog
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from typing import List, Dict, Union, Tuple, Optional, Literal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

load_dotenv()

from core.config.config import Config


class FeatureExtractor:
    """Class for extracting features and transforming them into vector representations(embeddings)."""
    def __init__(
        self,
        dataset_path: str,
        target_size: tuple,
        image_embeddings_path: str,
        text_embeddings_path: str,
        word_text_path: str,
        word_parents_path: str,
        bboxes_path: str,
        load_flag: bool = True
    ) -> None:
        self.word_text = []
        self.word_parents = []
        self.word_bboxes = []

        self.dataset_path = Path(dataset_path)
        self.target_size = target_size

        self.data_buffer: List[Dict] = self._process_text_image_pairs()
        
        self.image_embeddings_list = []
        self.text_embeddings_list = []
        
        if load_flag:
            pass
        else:
            self._process_image_text_pairs()
            

    def _process_text_image_pairs(self) -> List[Dict]:
        """Method for loading text-image pairs into general data buffer."""
        data_buffer = []

        # Check if dataset path exists
        if not self.dataset_path.exists():
            logger.error(f"Dataset path {self.dataset_path} does not exist")
            return data_buffer

        for letter_category in os.listdir(self.dataset_path):
            letter_dir = self.dataset_path / letter_category

            # Skip if it's not a directory
            if not letter_dir.is_dir():
                continue

            # Extract letter name from category folder name
            letter_name = letter_category.split('_')[0]

            # Construct annotation file path
            letter_annotation_path = letter_dir / f"{letter_name}_dataset.json"

            # Check if annotation file exists
            if not letter_annotation_path.exists():
                logger.warning(f"Annotation file {letter_annotation_path} not found")
                continue

            try:
                with open(letter_annotation_path, 'r', encoding='utf-8') as file:
                    letter_annotations = json.load(file)

                    # Process "with" objects (contains the letter)
                    if "with" in letter_annotations:
                        for obj in letter_annotations["with"]:
                            image_filename = obj.get("file_name")
                            relative_part = "/".join(image_filename.split("/")[-2:])
                            full_path = f"{letter_dir}/{relative_part}"
                            text_annotation = obj.get("label", "")

                            data_buffer.append({
                                "image_path": str(full_path),
                                "text": text_annotation,
                                "class": letter_name,
                                "contains": True
                            })

                    # Process "without" objects (doesn't contain the letter)
                    if "without" in letter_annotations:
                        for obj in letter_annotations["without"]:
                            image_filename = obj.get("file_name")
                            text_annotation = obj.get("label", "")

                            if image_filename:
                                image_path = str(letter_dir / image_filename)
                                data_buffer.append({
                                    "image_path": image_path,
                                    "text": text_annotation,
                                    "class": letter_name,
                                    "contains": False
                                })

            except (json.JSONDecodeError, KeyError, IOError) as e:
                logger.error(f"Error processing {letter_annotation_path}: {e}")
                continue

        logger.info(f"Loaded {len(data_buffer)} text-image pairs")
        return data_buffer

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Method for loading and preprocessing row image."""
        pass

    def _process_image_text_pairs(self) -> None:
        """Method for preprocessing image-text pairs."""
        for idx, data_point in enumerate(self.data_buffer):
            binary_image = self._preprocess_image(image_path=data_point["image_path"])


def main() -> None:
    """Main function."""
    config = Config()

    feature_extractor = FeatureExtractor(
        dataset_path=config.DATASET_PATH,
        target_size=config.TARGET_SIZE,
        image_embeddings_path=config.IMAGE_EMBEDDINGS_PATH,
        text_embeddings_path=config.TEXT_EMBEDDINGS_PATH,
        word_text_path=config.WORD_TEXT_PATH,
        word_parents_path=config.WORD_PARENTS_PATH,
        bboxes_path=config.WORD_BBOXES_PATH,
        load_flag=False
    )

    logger.info(feature_extractor.data_buffer[0])


if __name__ == "__main__":
    main()