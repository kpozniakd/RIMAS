import os
import gc
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
    """Interface for saving image embeddings."""
    def __init__(self, compression: str = "snappy") -> None:
        self.compression = compression

    @staticmethod
    def _to_table_format(batch_with_image_embeddings: List[Dict]) -> pd.DataFrame:
        """Convert List[Dict] -> pandas DataFrame."""
        rows = []
        for data_point in batch_with_image_embeddings:
            emb = data_point["image_embedding"]
            if isinstance(emb, np.ndarray):
                emb = emb.astype(np.float32).tolist()
            else:
                emb = np.asarray(emb, dtype=np.float32).tolist()
            rows.append({
                "id": int(data_point["id"]),
                "label": str(data_point["label"]),
                "image_path": str(data_point["image_path"]),
                "image_embedding": emb,
            })
        return pd.DataFrame(rows)

    def save_batch_parquet(
        self,
        batch_with_image_embeddings: List[Dict],
        output_dir: str,
        batch_idx: int
    ) -> None:
        """Save image embeddings to parquet files in batches."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not batch_with_image_embeddings:
            logger.warning(f"Empty batch #{batch_idx}; skip save.")
            return

        df = self._to_table_format(batch_with_image_embeddings)
        file_path = out_dir / f"part-{batch_idx:05d}.parquet"
        df.to_parquet(str(file_path), engine="pyarrow", compression=self.compression)

        del df
        gc.collect()
        logger.info(f"Image embeddings saved to {file_path} successfully.")
