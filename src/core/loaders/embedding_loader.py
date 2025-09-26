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


class EmbeddingsLoader:
    """Loader for image and text embeddings."""
    def __init__(self) -> None:
        pass

    def load_image_embeddings(
        self,
        image_embeddings_path: str
    ) -> List[Dict[str, List]]:
        """Method for loading image embeddings from parquet file."""
        image_embeddings = []
        if Path(image_embeddings_path).exists():
            try:
                image_df = pd.read_parquet(image_embeddings_path, engine="pyarrow")
                if "image_embedding" not in image_df.columns:
                    logger.error("image_embedding column not found in the file.")
                    return []

                image_embeddings = image_df["image_embedding"].apply(lambda x: np.array(x).tolist()).tolist()
                return image_embeddings

            except Exception as e:
                logger.error(f"Error loading image embeddings from {image_embeddings_path}: {e}")
                return []
        else:
            logger.warning(f"Image embeddings file not found: {image_embeddings_path}")
            return []
