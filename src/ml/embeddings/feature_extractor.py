import gc
import sys
import logging
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
        saver_service: EmbeddingsSaver,
        encoder_type: Literal["HOG", "SIFT", "Flatten"]
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.target_size = target_size
        self.encoder_type = encoder_type

        self.preprocessing_service = ImageProcessor()

        # Initialize data storage
        self.image_embeddings_list: List[Dict[str, str]] = []

        # Load data
        self.data_buffer = DataLoader.load_text_image_pairs(self.dataset_path)
        self.image_encoder = ImageEncoder(encoder_type=encoder_type)

    def prepare_and_save_text_image_pairs_with_batch(
        self,
        saver_service: EmbeddingsSaver,
        batch_size: int = 100,
        image_embeddings_path: str = None
    ) -> None:
        """Process text-image pairs in batches."""
        if not image_embeddings_path:
            raise ValueError("image_embeddings_path must be specified.")

        output_dir = Path(image_embeddings_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_number = len(self.data_buffer)
        if total_number == 0:
            raise ValueError("Dataset is empty !!!")

        for batch_start in tqdm(range(0, total_number, batch_size), desc="Embedding batches"):
            batch_end = min(total_number, batch_start + batch_size)
            batch = self.data_buffer[batch_start:batch_end]

            batch_with_image_embeddings: List[Dict] = []
            for offset, data_point in enumerate(batch):
                global_idx = batch_start + offset
                try:
                    binary_image, _ = self.preprocessing_service.preprocess_image(
                        data_point["image_path"], self.target_size
                    )
                    image_embedding = self.image_encoder.encode(binary_image)

                    batch_with_image_embeddings.append({
                        "image_embedding": image_embedding,
                        "label": data_point["text"],
                        "image_path": data_point["image_path"],
                        "id": global_idx
                    })
                except Exception as e:
                    logger.error(f"Error processing {data_point['image_path']}: {e}")

            saver_service.save_batch_parquet(
                batch_with_image_embeddings=batch_with_image_embeddings,
                output_dir=str(image_embeddings_path),
                batch_idx=batch_start // batch_size
            )

            del batch_with_image_embeddings
            gc.collect()

        logger.info(f"Done. Saved dataset under directory: {image_embeddings_path}")


if __name__ == "__main__":
    config = Config()
    saver_service = EmbeddingsSaver(
        compression=config.COMPRESSION
    )
    loader_service = EmbeddingsLoader()
    feature_extractor = FeatureExtractor(
        dataset_path=config.DATASET_PATH,
        target_size=config.TARGET_SIZE,
        image_embeddings_path=config.IMAGE_EMBEDDINGS_PATH,
        encoder_type=config.ENCODER_TYPE,
        saver_service=saver_service
    )

    # If you want to compute embeddings and save them in batches
    feature_extractor.prepare_and_save_text_image_pairs_with_batch(
        saver_service=saver_service,
        batch_size=config.BATCH_SIZE,
        image_embeddings_path=config.IMAGE_EMBEDDINGS_PATH
    )

    # If you want to load precomputed embeddings
    feature_extractor.image_embeddings_list = loader_service.load_image_embeddings_from_all_batches(
        image_embeddings_dir=config.IMAGE_EMBEDDINGS_PATH
    )
    print(feature_extractor.image_embeddings_list)