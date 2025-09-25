import os
import sys
import json
import string
import logging
from pathlib import Path
from skimage.feature import hog
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

load_dotenv()

from core.config.config import Config


class ImageProcessor:
    """Handles image processing operations."""
    @staticmethod
    def apply_morphological_operations(
        binary_image: np.ndarray, 
        kernel_size: Tuple[int, int] = (2, 2)
    ) -> np.ndarray:
        """Apply morphological operations to clean up binary image."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        morph = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
        return morph

    @staticmethod
    def preprocess_image(
        image_path: str,
        target_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess raw image."""
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Resize image
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale if needed
        if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
            grey = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        else:
            grey = resized_image

        # Denoise image
        denoised_image = cv2.fastNlMeansDenoising(
            grey, h=20, templateWindowSize=9, searchWindowSize=21
        )

        # Apply adaptive threshold
        binary_image = cv2.adaptiveThreshold(
            denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 31, 15
        )

        # Apply morphological operations
        binary_image = ImageProcessor.apply_morphological_operations(binary_image)

        return binary_image, grey

    @staticmethod
    def visualize_image(image: np.ndarray, window_name: str = "Image") -> None:
        """Visualize image in a window."""
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
            annotations = json.load(file)

        words_map = {
            Path(entry["filename"]).name: entry["label"]
            for entry in annotations.get("words", [])
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


class ImageEncoder:
    """Embed image data into vector representation."""
    def __init__(self) -> None:
        pass

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Method for encoding image data by flattening them into a vector."""
        return image.flatten()

    # def encode_image(self, image: np.ndarray) -> np.ndarray:
    #     """Method for encoding image data by using HOG (Histogram of Oriented Gradients) approach."""
    #     hog_features = hog(
    #         image,
    #         orientations=9,
    #         pixels_per_cell=(8,8),
    #         cells_per_block=(2,2),
    #         block_norm='L2-Hys',
    #         feature_vector=True,
    #         channel_axis=None
    #     )

    #     logger.debug(f"HOG features shape: {hog_features.shape}")
    #     return hog_features


class TextEncoder:
    """Embed text data into vector representation."""
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


class FeatureExtractor:
    """Extracts features and transforms them into vector representations (embeddings)."""
    def __init__(
        self,
        dataset_path: str,
        target_size: Tuple[int, int],
        image_embeddings_path: str,
        text_embeddings_path: str,
        load_flag: bool = True
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.target_size = target_size

        # Initialize data storage
        self.image_embeddings_list: List[np.ndarray] = []
        self.text_embeddings_list: List[np.ndarray] = []

        # Load data
        self.data_buffer = DataLoader.load_text_image_pairs(self.dataset_path)

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

        # Process data if not in load mode
        if not load_flag:
            self._process_image_text_pairs()
            self.save_embeddings(
                image_embeddings_path=image_embeddings_path,
                text_embeddings_path=text_embeddings_path
            )
        else:
            self._load_embeddings(
                image_embeddings_path=image_embeddings_path,
                text_embeddings_path=text_embeddings_path
            )

    def save_embeddings(self, image_embeddings_path: str, text_embeddings_path: str) -> None:
        """Method for saving embeddings."""
        try:
            image_embeddings_matrix = np.asarray(self.image_embeddings_list)
            text_embeddings_matrix = np.asarray(self.text_embeddings_list)

            Path(image_embeddings_path).parent.mkdir(parents=True, exist_ok=True)
            Path(text_embeddings_path).parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(image_embeddings_path, image=image_embeddings_matrix)
            np.savez_compressed(text_embeddings_path, text=text_embeddings_matrix)

            logger.info(f"Embeddings saved to {image_embeddings_path} and {text_embeddings_path} successfully.")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")

    def _load_embeddings(self, image_embeddings_path: str, text_embeddings_path: str) -> None:
        """Method for loading embeddings."""
        try:
            self.image_embeddings_list = np.load(f"{image_embeddings_path}")["image"]
            self.text_embeddings_list = np.load(f"{text_embeddings_path}")["text"]

            logger.info(f"Embeddings loaded from {image_embeddings_path} and {text_embeddings_path} successfully.")
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")

    def _process_image_text_pairs(self) -> None:
        """Preprocess image-text pairs."""
        for idx, data_point in enumerate(self.data_buffer[:]):
            try:
                binary_image, grey_image = ImageProcessor.preprocess_image(
                    data_point["image_path"], self.target_size
                )

                logger.info(f"Processing text-image pair: {idx}")

                image_embedding = self.image_encoder.encode_image(binary_image)
                text_embedding = self.text_encoder.encode_text(data_point["text"])

                self.image_embeddings_list.append(image_embedding)
                self.text_embeddings_list.append(text_embedding)

            except Exception as e:
                logger.error(f"Error processing {data_point['image_path']}: {e}")

    def get_data_summary(self) -> Dict:
        """Get summary of loaded data."""
        return {
            "total_samples": len(self.data_buffer),
            "classes": list(set(item["class"] for item in self.data_buffer)),
            "contains_count": sum(1 for item in self.data_buffer if item["contains"]),
            "without_count": sum(1 for item in self.data_buffer if not item["contains"])
        }


# TODO: If i need to embed only one image-text pair
class Inference:
    def __init__(self) -> None:
        pass


def main() -> None:
    """Main function to demonstrate FeatureExtractor functionality."""
    try:
        config = Config()

        feature_extractor = FeatureExtractor(
            dataset_path=config.DATASET_PATH,
            target_size=config.TARGET_SIZE,
            image_embeddings_path=config.IMAGE_EMBEDDINGS_PATH,
            text_embeddings_path=config.TEXT_EMBEDDINGS_PATH,
            load_flag=False
        )

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.exception("Detailed traceback:")


if __name__ == "__main__":
    main()