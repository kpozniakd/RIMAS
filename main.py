from src.core.loaders.kaggle_loader import KaggleLoader


def main():
    loader = KaggleLoader()
    loader.download_dataset("yiyueme/dataset-rimes")


if __name__ == "__main__":
    main()
