import os
from pathlib import Path
from src.core.settings import settings
from .interface_loader import IntefaceLoader


class KaggleLoader(IntefaceLoader):
    """Loader for downloading Kaggle datasets"""

    def __init__(self) -> None:
        os.environ.setdefault("KAGGLE_USERNAME", settings.KAGGLE_USERNAME)
        os.environ.setdefault("KAGGLE_KEY", settings.KAGGLE_KEY)
        from kaggle.api.kaggle_api_extended import KaggleApi  # Kaggle provider

        self.api = KaggleApi()
        self.api.authenticate()

    def download_dataset(self, dataset: str) -> Path:
        slug = dataset.rsplit("/", 1)[-1]
        dest = (settings.DATA_RAW_DIR / slug).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        self.api.dataset_download_files(
            dataset, path=str(dest), quiet=False, unzip=True
        )
        return dest
