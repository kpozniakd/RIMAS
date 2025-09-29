from typing import Tuple, List, Dict
import json
from pathlib import Path
import numpy as np
import cv2
import re
import logging
from ml.preprocessing.word_segmenter import WordSegmenter


logger = logging.getLogger(__name__)


class WordExporter:

    SYMBOL_NAMES = {
        "?": "questionmark",
        '"': "doublequote",
        "/": "forwardslash",
        ":": "colon",
    }

    def __init__(self, segmenter: WordSegmenter) -> None:
        self.segmenter = segmenter

    @staticmethod
    def _only_symbols(s: str) -> bool:
        return len(s) > 0 and all(ch in WordExporter.SYMBOL_NAMES for ch in s)

    @staticmethod
    def _symbols_to_names_joined(s: str) -> str:
        parts = [
            WordExporter.SYMBOL_NAMES[ch] for ch in s if ch in WordExporter.SYMBOL_NAMES
        ]
        return "_".join(parts) if parts else "word"

    @staticmethod
    def sanitize_label(label: str) -> str:
        lbl = str(label).lower()

        if WordExporter._only_symbols(lbl):
            return WordExporter._symbols_to_names_joined(lbl)

        cleaned = "".join(ch for ch in lbl if ch not in WordExporter.SYMBOL_NAMES)
        base = re.sub(r"[^a-z0-9_\-]+", "_", cleaned).strip("_")
        return base or "word"

    def export_from_dataframe(self, df, img_root: Path, out_dir: Path):
        """Save all words from dataframe"""
        out_dir.mkdir(parents=True, exist_ok=True)
        words_json = {"words": []}
        total = 0

        for row in df.itertuples(index=False):
            rel = getattr(row, "Filenames", None)
            text = getattr(row, "Contents", "")
            img_path = (img_root / rel) if img_root else Path(rel)
            pairs = self.segmenter.process_line(img_path, text)
            if pairs:
                saved = self.save_word_pairs(pairs, out_dir)
                words_json["words"].extend(
                    {"filename": str(p), "label": lbl} for p, lbl in saved
                )
                total += len(saved)

        with open(out_dir / "words" / "words.json", "w", encoding="utf-8") as jf:
            json.dump(words_json, jf, ensure_ascii=False, indent=2)

        print(f"Done! Word images saved: {total} Folder: {out_dir / 'words'}")

    def save_word_pairs(
        self,
        word_pairs: List[Tuple["np.ndarray", str]],
        out_dir: Path,
    ) -> List[Tuple[Path, str]]:
        words_dir = out_dir / "words" / "word"
        words_dir.mkdir(parents=True, exist_ok=True)

        saved: List[Tuple[Path, str]] = []
        counters: Dict[str, int] = {}

        for crop, label in word_pairs:
            base = self.sanitize_label(label)
            idx = counters.get(base, 0)
            counters[base] = idx + 1

            out_path = words_dir / f"{base}_{idx:05d}.png"
            cv2.imwrite(str(out_path), crop)
            saved.append((out_path.resolve(), label))

        return saved
