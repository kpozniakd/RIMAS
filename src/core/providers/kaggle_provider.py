from __future__ import annotations

from pathlib import Path
import os
from ..settings import settings

class KaggleProvider:
    def __init__(self):
        if not settings.KAGGLE_USERNAME or not settings.KAGGLE_KEY:
            raise RuntimeError("KAGGLE_USERNAME/KAGGLE_KEY not specified (add to .env).")
        os.environ["KAGGLE_USERNAME"] = settings.KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = settings.KAGGLE_KEY

        proj_cfg = Path(".kaggle/kaggle.json")
        if proj_cfg.exists():
            os.environ["KAGGLE_CONFIG_DIR"] = str(proj_cfg.parent.resolve())

        from kaggle.api.kaggle_api_extended import KaggleApi 

        self.api = KaggleApi()
        self.api.authenticate()

    def download_dataset(
        self,
        dataset: str,
        dest: Path | None = None,
        unzip: bool = True,
        force: bool = False,
    ) -> Path:
        if dest is None:
            slug = dataset.rsplit("/", 1)[-1]
            dest = settings.DATA_RAW_DIR / slug
        dest = Path(dest).resolve()
        dest.mkdir(parents=True, exist_ok=True)

        if not force and any(dest.iterdir()):
            print(f"Data already exist: {dest}")
            return dest

        print(f"Downloading {dataset} -> {dest} ...")
        self.api.dataset_download_files(dataset, path=str(dest), unzip=unzip, quiet=False)

        if unzip:
            for z in dest.glob("*.zip"):
                try:
                    z.unlink()
                except OSError:
                    pass

        print(f"Downloaded in {dest}")
        return dest
