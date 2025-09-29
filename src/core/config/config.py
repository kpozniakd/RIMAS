from dataclasses import dataclass
from typing import Tuple, Literal
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass
class Config:
    # Paths
    TRAIN_CSV_PATH: Path = (
        BASE_DIR / "data/raw/dataset-rimes/RIMES-2011-Lines/Train/train_labels.csv"
    )
    TEST_CSV_PATH: Path = (
        BASE_DIR / "data/raw/dataset-rimes/RIMES-2011-Lines/Test/test_labels.csv"
    )
    TRAIN_IMG_DIR: Path = (
        BASE_DIR / "data/raw/dataset-rimes/RIMES-2011-Lines/Train/Images"
    )
    TEST_IMG_DIR: Path = (
        BASE_DIR / "data/raw/dataset-rimes/RIMES-2011-Lines/Test/Images"
    )
    OUTPUT_PROCESSED_DIR: Path = BASE_DIR / "data/processed"

    IMAGE_EMBEDDINGS_PATH: Path = (
        BASE_DIR / "data/processed/words/weights/image_embeddings.parquet"
    )
    TEXT_EMBEDDINGS_PATH: Path = (
        BASE_DIR / "data/processed/words/weights/text_embeddings.parquet"
    )
    WORD_TEXT_PATH: Path = BASE_DIR / "data/processed/weights/word_text"
    WORD_PARENTS_PATH: Path = BASE_DIR / "data/processed/weights/word_parents"
    WORD_BBOXES_PATH: Path = BASE_DIR / "data/processed/weights/word_bboxes"
    DATASET_PATH: Path = BASE_DIR / "data/processed/words"

    # Image / text sizes
    IMAGE_SIZE: int = 384
    TARGET_SIZE: Tuple[int, int] = (400, 150)
    MAX_TEXT_LENGTH: int = 128

    # Defaults
    DEFAULT_TOP_N: int = 30
    DEFAULT_IMG_COUNT: int = 5
    DEFAULT_KERNEL = (
        29,
        3,
    )  # rectangular kernel for dilation (merged letters into words)
    DEFAULT_MIN_AREA_RATIO = 0.006  # min. fraction of image area for word outline
    ENCODER_TYPE: Literal["HOG", "SIFT", "Flatten"] = "SIFT"

    # Quality thresholds
    BLANK_STD_THRESHOLD: float = 5
    SHARP_LAPLACIAN_THRESHOLD: float = 300
    NOISE_STD_MEAN_THRESHOLD: float = 0.7
    CONTRAST_DIFF_THRESHOLD: float = 30

    # Window parameters
    TARGET_HEIGHT: int = 64
    WINDOW_WIDTH: int = 30
    STRIDE_SIZE: int = 10


# if __name__ == "__main__":
# config = Config()
# print(config.DATASET_PATH)
# print(config.TRAIN_CSV_PATH)
