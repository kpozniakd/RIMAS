import sys
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Literal

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.config.config import Config
from core.loaders.data_loader import DataLoader
from core.loaders.embedding_loader import EmbeddingsLoader
from utils.saving_utils.save_embeddings import EmbeddingsSaver
from ml.preprocessing.image_preprocessing import ImageProcessor
from ml.embeddings.image_encoder import ImageEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Main class for preprocessing entier dataset and extracting image and text embeddings."""
    def __init__(
        self,
        dataset_path: str,
        target_size: Tuple[int, int],
        image_embeddings_path: str,
        encoder_type: Literal["HOG", "SIFT", "Flatten"],
        load_flag: bool = True
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.target_size = target_size
        self.encoder_type = encoder_type

        self.loader_service = EmbeddingsLoader()
        self.saver_service = EmbeddingsSaver()
        self.preprocessing_service = ImageProcessor()

        # Initialize data storage
        self.image_embeddings_list: List[Dict[str, str]] = []

        # Load data
        self.data_buffer = DataLoader.load_text_image_pairs(self.dataset_path)

        self.image_encoder = ImageEncoder(encoder_type=encoder_type)

        if load_flag:
            self.image_embeddings_list = self.loader_service.load_image_embeddings(image_embeddings_path=image_embeddings_path)
        else:
            self.prepare_text_image_pairs()
            self.saver_service.save_image_embeddings(
                image_embeddings_path=image_embeddings_path,
                image_embeddings_list=self.image_embeddings_list,
                data_buffer=self.data_buffer
                )

    def prepare_text_image_pairs(self) -> None:
        """Preprocess image-text pairs."""
        for idx, data_point in enumerate(tqdm(self.data_buffer[:])):
            try:
                binary_image, grey_image = self.preprocessing_service.preprocess_image(
                    data_point["image_path"], self.target_size
                )
                image_embedding = self.image_encoder.encode(binary_image)

                self.image_embeddings_list.append({
                    "image_embedding": image_embedding,
                    "label": data_point["text"],
                    "image_path": data_point["image_path"]
                })
            except Exception as e:
                logger.error(f"Error processing {data_point['image_path']}: {e}")


# if __name__ == "__main__":
#     config = Config()
#     feature_extractor = FeatureExtractor(
#         dataset_path=config.DATASET_PATH,
#         target_size=config.TARGET_SIZE,
#         image_embeddings_path=config.IMAGE_EMBEDDINGS_PATH,
#         encoder_type=config.ENCODER_TYPE,
#         load_flag=True
#     )

#     print(feature_extractor.image_embeddings_list[:1])
