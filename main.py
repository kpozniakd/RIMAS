from pathlib import Path
from src.core.providers.kaggle_provider import KaggleProvider

def main():
    provider = KaggleProvider()
    provider.download_dataset("yiyueme/dataset-rimes")
    

if __name__ == "__main__":
    main()


