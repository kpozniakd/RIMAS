from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
    KAGGLE_KEY = os.getenv("KAGGLE_KEY")
    DATA_RAW_DIR = Path(os.getenv("DATA_RAW_DIR", "src/data/raw"))

settings = Settings()
