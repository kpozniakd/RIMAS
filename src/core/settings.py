from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=False)


@dataclass(frozen=True)
class Settings:
    DATA_RAW_DIR: Path
    KAGGLE_USERNAME: str
    KAGGLE_KEY: str


def _build_settings() -> Settings:
    print("this is orig")
    data_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw")).resolve()
    user = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not user or not key:
        raise RuntimeError("KAGGLE_USERNAME/KAGGLE_KEY not specified (add to .env).")
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(DATA_RAW_DIR=data_dir, KAGGLE_USERNAME=user, KAGGLE_KEY=key)


settings = _build_settings()
