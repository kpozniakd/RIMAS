import cv2
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from transformers import TrOCRProcessor
import os
from PIL import Image
from skimage.morphology import skeletonize
import numpy as np
from tqdm import tqdm


def plot_text_length_distribution(df: pd.DataFrame) -> None:
    """Plot histogram of line lengths in characters."""
    df["Len of content"] = df["Contents"].apply(len)
    plt.figure(figsize=(10, 6))
    plt.hist(df["Len of content"], bins=30, edgecolor="black")
    plt.title("Distribution of text string lengths (in chars)")
    plt.xlabel("Number of chars in a line")
    plt.ylabel("Number of examples")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def word_line_distribution(single_count: int, multi_count: int) -> None:
    """Visual comparison of single vs multi-word lines."""
    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Single Word", "Multiple Words"],
        [single_count, multi_count],
        color=["skyblue", "salmon"],
    )
    plt.title("Single Word vs Multiple Words in Text")
    plt.ylabel("Number of Images")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def analyze_word_frequency(df: pd.DataFrame, top_n: int = 30) -> None:
    """Display most frequent words in the dataset."""
    word_freq = Counter()
    for text in df["Contents"].astype(str):
        word_freq.update(text.split())
    words, freqs = zip(*word_freq.most_common(top_n))

    plt.figure(figsize=(14, 6))
    plt.bar(words, freqs)
    plt.title(f"Top {top_n} Most Frequent Words")
    plt.xticks(rotation=45)
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_sample_images(df: pd.DataFrame, img_dir: str, n: int = 5) -> None:
    """Visualize first N image-label pairs."""
    plt.figure(figsize=(6, 2 * n))
    for i in range(n):
        path = os.path.join(img_dir, df.iloc[i]["Filenames"])
        label = df.iloc[i]["Contents"]
        img = Image.open(path)
        plt.subplot(n, 1, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"'{label}'")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# def analyze_token_lengths(
#     df: pd.DataFrame, processor: TrOCRProcessor, max_length: int = 128
# ) -> None:
#     """Plot distribution of tokenized sequence lengths."""
#     token_lengths = df["Contents"].apply(
#         lambda t: len(processor.tokenizer(t).input_ids)
#     )
#     over_limit = (token_lengths > max_length).sum()

#     plt.hist(token_lengths, bins=30, edgecolor="black")
#     plt.title("Distribution of tokenized text lengths")
#     plt.xlabel("Token count")
#     plt.ylabel("Number of samples")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     print(
#         f"Samples over max_length ({max_length}): {over_limit} / {len(df)} ({over_limit / len(df) * 100:.2f}%)"
#     )


# def analyze_word_count_type(df: pd.DataFrame) -> None:
#     """Analyze lines that contain one word vs multiple."""
#     df["word_count"] = df["Contents"].apply(lambda x: len(str(x).split()))
#     single = (df["word_count"] == 1).sum()
#     multiple = (df["word_count"] > 1).sum()
#     word_line_distribution(single, multiple)


# import matplotlib.pyplot as plt


# def load_and_process_images_from_dir(directory, exts=(".png", ".jpg", ".jpeg")):
#     """Scan a directory for images, binarize each, estimate stroke thickness,
#     and return (filenames, grayscale images, thicknesses)."""
#     filenames, images, thicknesses = [], [], []

#     for fname in tqdm(os.listdir(directory), desc="Check images"):
#         if not fname.lower().endswith(exts):
#             continue
#         path = os.path.join(directory, fname)
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             continue
#         _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         thickness = estimate_stroke_thickness(binary)
#         thicknesses.append(thickness)
#         images.append(img)
#         filenames.append(fname)

#     return filenames, images, thicknesses


# def estimate_stroke_thickness(binary_img):
#     """Estimate average stroke thickness as (foreground area / skeleton length);
#     auto-inverts if foreground is light on dark."""
#     if binary_img.mean() > 127:
#         binary_img = cv2.bitwise_not(binary_img)
#     binary = (binary_img > 0).astype(np.uint8)
#     skeleton = skeletonize(binary)
#     area_original = np.sum(binary)
#     area_skeleton = np.sum(skeleton)
#     if area_skeleton == 0:
#         return 0
#     return area_original / area_skeleton


# def plot_top_thickest_lines(img_dir: str, n: int = 5) -> None:
#     """Show N images with thickest lines"""
#     filenames, images, thicknesses = load_and_process_images_from_dir(img_dir)
#     df_lines = pd.DataFrame(
#         {"filename": filenames, "image": images, "thickness": thicknesses}
#     )
#     df_lines["thickness"] = df_lines["image"].apply(estimate_stroke_thickness)
#     df_top = df_lines.sort_values("thickness", ascending=False).head(n)

#     plt.figure(figsize=(3 * n, 4))
#     for i, row in enumerate(df_top.itertuples(), 1):
#         plt.subplot(1, n, i)
#         plt.imshow(row.image, cmap="gray")
#         plt.title(f"{row.thickness:.2f}")
#         plt.axis("off")
#     plt.tight_layout()
#     plt.show()
#     Q1 = df_lines["thickness"].quantile(0.25)
#     Q3 = df_lines["thickness"].quantile(0.75)
#     IQR = Q3 - Q1

#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     outliers = (df_lines["thickness"] < lower_bound) | (
#         df_lines["thickness"] > upper_bound
#     )

#     num_outliers = outliers.sum()
#     total = len(df_lines)
#     percentage_outliers = (num_outliers / total) * 100

#     print(f"Number of outliers: {num_outliers} з {total}")
#     print(f"% of outliers: {percentage_outliers:.2f}%")


# def compare_croped_img(words_df, words_df_normalized, i: int):
#     """Side-by-side visualization of original vs cropped/normalized word image for index i."""
#     fig, axes = plt.subplots(1, 2)
#     fig.patch.set_facecolor("lightgray")

#     axes[0].imshow(words_df["image"].iloc[i], cmap="gray")
#     axes[0].set_title("Original")
#     axes[0].axis("off")

#     axes[1].imshow(words_df_normalized["image"].iloc[i], cmap="gray")
#     axes[1].set_title("Croped")
#     axes[1].axis("off")

#     plt.tight_layout()
#     plt.show()


def check_images_with_duplcated_text(df: pd.DataFrame, img_dir: str):
    duplicated_texts = df[df.duplicated("Contents", keep=False)]
    example_texts = (
        duplicated_texts["Contents"]
        .value_counts()
        .loc[lambda x: x > 1]
        .head(1)
        .index.tolist()
    )
    for text in example_texts:
        subset = duplicated_texts[duplicated_texts["Contents"] == text]
        print(f"\nText: '{text}' - {len(subset)}")
        plt.figure(figsize=(15, 3))
        for i, (_, row) in enumerate(subset.iterrows()):
            if i >= 5:
                break
            img_path = os.path.join(img_dir, row["Filenames"])
            img = Image.open(img_path)
            plt.subplot(1, 5, i + 1)
            plt.imshow(img, cmap="gray")
            plt.title(f"{row['Filenames']}", fontsize=8)
            plt.axis("off")
        plt.suptitle(f'Images with the same text:\n"{text}"', fontsize=12)
        plt.tight_layout()
        plt.show()
