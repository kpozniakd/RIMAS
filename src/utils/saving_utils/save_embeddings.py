import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingsSaver:
    """Interface for saving image and text embeddings."""
    def __init__(self) -> None:
        pass

    def save_image_embeddings(
        self,
        image_embeddings_list: List[np.ndarray],
        data_buffer: List[Dict],
        image_embeddings_path: str
    ) -> List[Dict[str, List]]:
        """Method for saving image embeddings in parquet format."""
        if Path(image_embeddings_path).exists():
            image_data = [{
               "image_embedding": image_embedding,
               "label": data_point["text"],
               "image_path": data_point["image_path"]
            } for image_embedding, data_point in zip(image_embeddings_list, data_buffer)]
            image_df = pd.DataFrame(image_data)

            Path(image_embeddings_path).parent.mkdir(parents=True, exist_ok=True)

            image_df.to_parquet(image_embeddings_path, engine='pyarrow', compression='snappy')
            logger.info(f"Image embeddings saved to {image_embeddings_path} successfully.")
        else:
            logger.warning(f"Image embeddings file not found: {image_embeddings_path}")
