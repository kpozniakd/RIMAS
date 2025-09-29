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

    def load_image_embeddings_from_all_batches(
        self,
        image_embeddings_dir: str,
        sort_by_id: bool = True,
        as_dataframe: bool = True,
    ):
        """
        Load all image embeddings from all batches into a sigle instance.
        """
        directory = Path(image_embeddings_dir)
        if not directory.exists():
            logger.warning(f"Image embeddings directory not found: {directory}")
            return pd.DataFrame() if as_dataframe else []

        files = sorted(directory.glob("*.parquet"))
        if not files:
            logger.warning(f"No parquet files found under {directory}")
            return pd.DataFrame() if as_dataframe else []

        frames: List[pd.DataFrame] = []
        for file in files:
            try:
                df = pd.read_parquet(file, engine="pyarrow")
                frames.append(df)
            except Exception as e:
                logger.error(f"Failed to read {file}: {e}")

        if not frames:
            return pd.DataFrame() if as_dataframe else []

        full = pd.concat(frames, ignore_index=True)

        if sort_by_id and "id" in full.columns:
            full = full.sort_values("id").reset_index(drop=True)

        if as_dataframe:
            return full

        if "image_embedding" in full.columns:
            full["image_embedding"] = full["image_embedding"].apply(
                lambda x: np.asarray(x, dtype=np.float32).tolist()
            )
        return full.to_dict("records")
