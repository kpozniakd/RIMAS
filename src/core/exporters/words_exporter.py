from src.core.config.config import Config
from typing import Tuple
import json
from pathlib import Path
import numpy as np
import cv2


class WordExporter:
    def __init__(
        self,
        kernel: Tuple[int, int] = Config.DEFAULT_KERNEL,
        min_area_ratio: float = Config.DEFAULT_MIN_AREA_RATIO,
    ) -> None:
        self.kernel = kernel
        self.min_area_ratio = min_area_ratio

    def export_from_dataframe(self, df, img_root: Path, out_dir: Path):
        """Save all words from dataframe"""
        out_dir.mkdir(parents=True, exist_ok=True)
        words_json = {"words": []}
        total = 0

        for row in df.itertuples(index=False):
            rel = getattr(row, "Filenames", None)
            text = getattr(row, "Contents", "")
            if not rel:
                continue

            img_path = (img_root / rel) if img_root else Path(rel)
            pairs = self.process_line(img_path, text, out_dir)
            if pairs:
                words_json["words"].extend(
                    {"filename": str(p), "label": lbl} for p, lbl in pairs
                )
                total += len(pairs)

        with open(out_dir / "words.json", "w", encoding="utf-8") as jf:
            json.dump(words_json, jf, ensure_ascii=False, indent=2)

        print(f"Done! Word images saved: {total} Folder: {out_dir / 'words'}")
