import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image


def analyze_image_dimensions(df: pd.DataFrame, img_dir: str) -> None:
    """Plot image width vs height and aspect ratios."""

    def get_size(name):
        try:
            with Image.open(os.path.join(img_dir, name)) as img:
                return img.size
        except:
            return (None, None)

    df[["width", "height"]] = df["Filenames"].apply(get_size).apply(pd.Series)
    df["aspect_ratio"] = df["width"] / df["height"]

    plt.scatter(df["width"], df["height"], alpha=0.3)
    plt.title("Width vs Height of images")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.grid(True)
    plt.show()


def check_channels(file_name: str, img_dir: str) -> str:
    """Get channel mode (e.g. RGB, L) of image."""
    with Image.open(os.path.join(img_dir, file_name)) as img:
        return img.mode


def analyze_channel_distribution(df: pd.DataFrame, img_dir: str) -> pd.Series:
    """Return counts of image modes across dataset."""
    df["image_mode"] = df["Filenames"].apply(lambda f: check_channels(f, img_dir))
    return df["image_mode"].value_counts()
