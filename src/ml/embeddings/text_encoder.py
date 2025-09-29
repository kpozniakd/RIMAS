import sys
import string
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


class TextEncoder:
    """Interface for encoding text data into vector representation."""
    def __init__(self) -> None:
        self.alphabet = self._create_alphabet()

    def _create_alphabet(self) -> str:
        """Method for creating alphabet for text encoding algorithm."""
        alphabet = string.ascii_lowercase + string.ascii_uppercase
        digits = string.digits
        punctuation = string.punctuation
        french_accents = "àâäçéèêëîïôöùûüÿœæ"
        french_alphabet = alphabet + digits + punctuation + french_accents

        logger.debug(f"Alphabets: {french_alphabet}")

        return french_alphabet

    def encode_text(self, text: str) -> np.ndarray:
        """Method for encoding text data by using Bag-Of-Letters approach."""
        bag_of_letters = np.zeros(len(self.alphabet), dtype=int)

        # Create a mapping from character to index
        char_to_index = {char: i for i, char in enumerate(self.alphabet)}

        for char in text:
            if char in char_to_index:
                index = char_to_index[char]
                bag_of_letters[index] += 1

        logger.debug(f"Bag-Of-Letters features shape: {bag_of_letters.shape}")

        return bag_of_letters
