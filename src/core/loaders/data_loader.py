import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Literal

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and processing of text-image pairs."""
    @staticmethod
    def load_text_image_pairs(dataset_path: Path) -> List[Dict]:
        """Load text-image pairs from dataset directory."""
        data_buffer: list = []

        if not dataset_path.exists():
            logger.error(f"Dataset path {dataset_path} does not exist")
            return data_buffer

        annotation_path = dataset_path / "words.json"

        with open(annotation_path, "r", encoding="utf-8") as file:
            annotation = json.load(file)

        words_map = {
            Path(entry["filename"]).name: entry["label"]
            for entry in annotation.get("words", [])
        }
        word_dir = dataset_path / "word"
        for image_name in os.listdir(word_dir):
            if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            if image_name in words_map:
                image_path = word_dir / image_name
                data_buffer.append({
                    "image_path": str(image_path),
                    "text": words_map[image_name]
                })

        logger.info(f"Loaded {len(data_buffer)} text-image pairs")
        return data_buffer
